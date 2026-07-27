"""RAG 自动化质量评估骨架（基于 Ragas）。

设计目标：你只准备「黄金问答集」，其余（调生产检索管线生成答案与召回上下文、
调 Ragas 算指标、出报告）本脚本全部完成。

- 复用 mildoc_wxkf/core 的检索 / 生成管线（与线上完全一致，分数才代表生产行为）。
- 评估指标：
    faithfulness     答案是否忠于检索内容（不胡编）
    answer_relevancy 答案是否切题
    context_precision 召回上下文是否相关（需 reference）
    context_recall    召回上下文是否覆盖标准答案（需 reference）

运行：
    cd mildoc_eval
    pip install -r requirements.txt
    python run_eval.py                 # 默认读 datasets/golden_set.jsonl
    python run_eval.py 路径/自定义.jsonl

评估裁判(judge)模型在 mildoc_eval/.env 的 JUDGE_MODELS 中配置（逗号分隔多个模型，
按问题轮询分摊免费额度）；检索/生成仍走 mildoc_wxkf/.env 的生产模型。
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

# ============ 路径与配置加载（必须在 import wxkf 模块之前）============
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # mildoc_llamaindex/
WXKF = ROOT / "mildoc_wxkf"            # 含 config/ core/ memory/ logger/ extensions
sys.path.insert(0, str(WXKF))          # 让 `config` `core` `memory` `logger` `extensions` 可导入

# 加载 wxkf 的 .env（含 Milvus / Redis / LLM / Embedding 配置），覆盖已存在变量。
# 之后 config.config 在 import 时再次 load_dotenv() 不会覆盖这里已写入的值。
load_dotenv(WXKF / ".env", override=True)

# 加载评估专属配置（mildoc_eval/.env）：仅含 JUDGE_MODELS 等裁判模型设置，
# 不覆盖生产管线使用的 LLM 配置（仍由 mildoc_wxkf/.env 提供）。
load_dotenv(HERE / ".env", override=True)

from config.config import Config  # noqa: E402  （import 顺序依赖上方 path/dotenv 设置）

# ============ Ragas ============
from ragas import EvaluationDataset, evaluate, RunConfig  # noqa: E402
from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
try:  # 限流器：部分 langchain_core 版本路径不同，取不到则退化为不限流
    from langchain_core.rate_limiters import SimpleRateLimiter
except Exception:  # noqa: BLE001
    SimpleRateLimiter = None


# ============ 复用生产检索管线 ============
def get_pipeline():
    """根据 Config.RETRIEVER_MODE 选择 default / hierarchical 检索管线。"""
    mode = (Config.RETRIEVER_MODE or "default").lower()
    if mode == "hierarchical":
        from core.hierarchical_retriever import HierarchicalRetrievalPipeline as Cls
    else:
        from core.default_retriever import DefaultRetrievalPipeline as Cls
    print(f">>> 检索模式：{mode}")
    return Cls()


def run_one(pipeline, question: str):
    """用生产管线对单条问题做检索 + 生成，返回 (答案全文, 召回上下文列表)。"""
    qe = pipeline.get_query_engine()
    resp = qe.query(question)
    answer = str(resp)  # 流式响应：转 str 会阻塞直至生成完成，取全文
    contexts = [n.get_content() for n in resp.source_nodes]
    return answer, contexts


# ============ Ragas 模型 ============
# 检索/生成仍走生产管线（Config，复用 mildoc_wxkf/.env 的生产模型）。
# 评估裁判(judge)模型独立配置：读取 mildoc_eval/.env 的 JUDGE_MODELS（逗号分隔，多个模型按问题轮询）。
# 向量化复用与生产索引【完全相同】的 LlamaIndex OpenAIEmbedding 客户端：
#   1) 规避 LangChain OpenAIEmbeddings 走 OpenAI 兼容接口时，text-embedding-v4 报
#      "contents is neither str nor list of str" 的 400；
#   2) 保证评估与索引处于同一向量空间，answer_relevancy 的相似度才有意义。
class _LlamaEmbeddingsAdapter(Embeddings):
    """把 LlamaIndex 的 OpenAIEmbedding 适配成 LangChain Embeddings 接口，供 Ragas 使用。"""

    def __init__(self, llama_model):
        super().__init__()
        self._model = llama_model

    def embed_documents(self, texts):
        return self._model.get_text_embedding_batch(list(texts))

    def embed_query(self, text):
        return self._model.get_text_embedding(text)


def build_embeddings():
    from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
    llama_emb = LlamaOpenAIEmbedding(
        model_name=Config.ENBEDDING_MODEL,      # "text-embedding-v4"，与生产一致
        dimensions=Config.MILVUS_VECTOR_DIM,    # 与 Milvus 维度对齐
        api_key=Config.OPENAI_API_KEY,          # 对应 .env 的 LLM_EMBEDDING_API_KEY
        api_base=Config.OPENAI_BASE_URL,        # 对应 .env 的 LLM_EMBEDDING_BASE_URL
        embed_batch_size=10,                     # 百炼单次 embedding 上限 10 条
        max_retries=3,                           # 吸收偶发超时
        timeout=120,                            # 单次 embedding 上限 120s
    )
    return LangchainEmbeddingsWrapper(_LlamaEmbeddingsAdapter(llama_emb))


def build_judge_llm(model_name: str):
    """为单个评估裁判模型构建 LLM 包装（默认复用同一账户的 api_key / base_url）。"""
    api_key = os.getenv("JUDGE_API_KEY") or Config.LLM_API_KEY
    base_url = os.getenv("JUDGE_BASE_URL") or Config.LLM_BASE_URL
    # 限流：免费额度模型 RPM 很低，Ragas 默认高并发会瞬间打满触发 429。
    # 用 SimpleRateLimiter 把请求均匀化；速率由 JUDGE_RPS 调整（默认保守值）。
    rps = float(os.getenv("JUDGE_RPS", "0.34"))
    rate_limiter = SimpleRateLimiter(requests_per_second=rps) if SimpleRateLimiter else None
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_retries=5,      # 偶发 429/超时，指数退避重试
            timeout=120,        # 单次调用上限 120s，避免无限挂起
            rate_limiter=rate_limiter,
        )
    )


def get_judge_models() -> list:
    raw = os.getenv("JUDGE_MODELS", "").strip()
    models = [m.strip() for m in raw.split(",") if m.strip()]
    # 兜底：没配 JUDGE_MODELS 时退化为生产模型单例（行为与改动前一致）
    return models or [Config.LLM_MODEL_NAME]

# ============ 数据集加载 ============
def load_dataset(path: Path) -> list:
    if not path.exists():
        raise SystemExit(
            f"找不到数据集: {path}\n请参考 datasets/README.md 准备 "
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


# ============ 主流程 ============
data_set_name = 'golden_set2.jsonl'
def main():
    # 数据集目录
    dataset_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE / "datasets" / f"{data_set_name}")
    )
    # 结果输出目录
    out_path = HERE / "eval_results2.xlsx"

    records = load_dataset(dataset_path)
    if not records:
        raise SystemExit("数据集为空，请先准备问题。")

    pipeline = get_pipeline()
    embeddings = build_embeddings()
    judge_models = get_judge_models()
    print(f">>> 评估裁判模型（按问题轮询）: {judge_models}")

    # 向量化(embeddings)自检：answer_relevancy 依赖它，调用失败时整列会成 NaN 且无报错。
    # 生产检索走 LlamaIndex 自有封装，不代表这里 LangChain 的 OpenAI 兼容客户端可用，
    # 因此单独探活，失败直接给出明确错误而不是闷成 nan。
    # try:
    #     _v = embeddings.embed_query("测试 embedding 是否可用")
    #     if not _v:
    #         raise ValueError("embedding 返回空向量")
    #     print(f">>> embeddings 自检通过（维度={len(_v)}）")
    # except Exception as e:
    #     raise SystemExit(
    #         f"[ embeddings 不可用 ] answer_relevancy 将全部为 NaN。请检查 mildoc_wxkf/.env 的 "
    #         f"LLM_EMBEDDING_MODEL_NAME / LLM_EMBEDDING_BASE_URL / LLM_EMBEDDING_API_KEY。"
    #         f"\n原始错误：{type(e).__name__}: {e}"
    #     )

    # EVAL_LIMIT>0 时只跑前 N 条（用于快速试跑），默认 0 = 全部
    limit = int(os.getenv("EVAL_LIMIT", "0") or 0)

    # 1) 跑生产管线，生成答案 + 召回上下文
    eval_rows = []
    for i, rec in enumerate(records, 1):
        q = rec.get("user_input") or rec.get("question")
        if not q:
            print(f"[{i}] 跳过：缺少 user_input 字段 -> {rec}")
            continue
        print(f"[{i}/{len(records)}] 检索+生成: {q[:40]} ...")
        answer, contexts = run_one(pipeline, q)
        # 若数据集自带 response 则用自带（你手动准备的「系统答案」），否则用管线生成的
        response_text = rec.get("response") or answer
        eval_rows.append(
            {
                "user_input": q,
                "response": response_text,
                "reference": rec.get("reference", ""),
                "retrieved_contexts": contexts,
            }
        )
        if limit and len(eval_rows) >= limit:
            print(f"\n[EVAL_LIMIT={limit}] 试跑达到上限，停止。")
            break

    # 2) 按问题轮询分片到各裁判模型（每个问题的子指标由同一模型打分，保证内部一致）
    n_models = len(judge_models)
    shards = [[] for _ in range(n_models)]
    for idx, row in enumerate(eval_rows):
        shards[idx % n_models].append(row)

    # 仅当有标准答案(reference)时才评估依赖 reference 的指标
    has_ref = all(r["reference"] for r in eval_rows)
    metrics = [faithfulness, answer_relevancy]
    if has_ref:
        metrics += [context_precision, context_recall]
    else:
        print("\n[提示] 部分/全部记录缺少 reference（标准答案），"
              "已跳过 context_precision / context_recall，仅评估 faithfulness / answer_relevancy。")

    # 3) 每个分片用对应裁判模型单独 evaluate，再合并明细 + 加权汇总
    # RunConfig：max_workers=1 串行化所有 LLM/embedding 调用，避免瞬时并发打满免费 RPM；
    # max_retries 与 ChatOpenAI 的退避重试叠加，偶发 429 也能恢复。
    run_config = RunConfig(max_workers=1, timeout=120, max_retries=5)
    shard_dfs = []
    for mi, model in enumerate(judge_models):
        shard = shards[mi]
        if not shard:
            continue
        print(f"\n===== 裁判模型 [{model}] 评估 {len(shard)} 条 =====")
        llm = build_judge_llm(model)
        ds = EvaluationDataset.from_list(shard)
        result = evaluate(
            dataset=ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
        )
        print(result)
        df = result.to_pandas()
        df["_judge_model"] = model
        shard_dfs.append(df)

    if not shard_dfs:
        raise SystemExit("没有可用于评估的样本。")

    combined = pd.concat(shard_dfs, ignore_index=True)
    print("\n===== 跨模型汇总（按分片大小加权均值）=====")
    summary_rows = []
    for m in metrics:
        name = m.name
        if name in combined.columns:
            mean_val = combined[name].mean()
            print(f"  {name:18s}: {mean_val:.4f}")
            summary_rows.append({"指标": name, "加权均值": round(float(mean_val), 4)})
    # 按裁判模型分别汇总（多模型轮询时有意义）
    if len(judge_models) > 1 and "_judge_model" in combined.columns:
        for model in judge_models:
            sub = combined[combined["_judge_model"] == model]
            if sub.empty:
                continue
            print(f"  [{model}]")
            for m in metrics:
                name = m.name
                if name in sub.columns:
                    print(f"    {name:18s}: {sub[name].mean():.4f}")

    # ---- 保存为 xlsx（CSV 分隔符难用，改用 Excel 原生格式）----
    # retrieved_contexts 是 list，直接写 Excel 会报错，先拼成可读文本。
    if "retrieved_contexts" in combined.columns:
        combined["retrieved_contexts"] = combined["retrieved_contexts"].apply(
            lambda c: "\n\n----\n\n".join(c) if isinstance(c, (list, tuple)) else c
        )

    try:
        from openpyxl import Workbook  # noqa: F401  确保引擎可用
        summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
        # 多模型时附上分模型汇总，方便对比
        if len(judge_models) > 1 and "_judge_model" in combined.columns:
            per_model = []
            for model in judge_models:
                sub = combined[combined["_judge_model"] == model]
                if sub.empty:
                    continue
                row = {"裁判模型": model, "样本数": len(sub)}
                for m in metrics:
                    name = m.name
                    if name in sub.columns:
                        row[name] = round(float(sub[name].mean()), 4)
                per_model.append(row)
            if per_model:
                summary_df = pd.concat(
                    [summary_df, pd.DataFrame([{}]), pd.DataFrame(per_model)],
                    ignore_index=True,
                )
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="明细", index=False)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name="汇总", index=False)
        print(f"\n已保存 Excel 报告（明细 + 汇总两个 sheet）到: {out_path}")
    except ImportError:
        # 没装 openpyxl 时优雅降级为 CSV，并提示安装
        csv_path = out_path.with_suffix(".csv")
        combined.to_csv(csv_path, index=False)
        print(f"\n[提示] 未安装 openpyxl，已降级输出 CSV: {csv_path}")
        print("        安装 xlsx 支持：pip install openpyxl")
    except Exception as e:  # 保存失败不影响结果展示
        print(f"保存 Excel 失败（不影响结果）: {e}")


if __name__ == "__main__":
    main()

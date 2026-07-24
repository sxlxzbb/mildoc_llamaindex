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
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402


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
def build_embeddings():
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=Config.ENBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY,       # 对应 .env 的 LLM_EMBEDDING_API_KEY
            base_url=Config.OPENAI_BASE_URL,     # 对应 .env 的 LLM_EMBEDDING_BASE_URL
            chunk_size=10,                        # 百炼单次 embedding 上限 10 条
            dimensions=Config.MILVUS_VECTOR_DIM,
        )
    )


def build_judge_llm(model_name: str):
    """为单个评估裁判模型构建 LLM 包装（默认复用同一账户的 api_key / base_url）。"""
    api_key = os.getenv("JUDGE_API_KEY") or Config.LLM_API_KEY
    base_url = os.getenv("JUDGE_BASE_URL") or Config.LLM_BASE_URL
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
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
            f"找不到数据集: {path}\n请参考 datasets/README.md 准备 golden_set.jsonl"
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


# ============ 主流程 ============
def main():
    dataset_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE / "datasets" / "golden_set.jsonl")
    )
    out_path = HERE / "eval_results.csv"

    records = load_dataset(dataset_path)
    if not records:
        raise SystemExit("数据集为空，请先准备问题。")

    pipeline = get_pipeline()
    embeddings = build_embeddings()
    judge_models = get_judge_models()
    print(f">>> 评估裁判模型（按问题轮询）: {judge_models}")

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
        )
        print(result)
        df = result.to_pandas()
        df["_judge_model"] = model
        shard_dfs.append(df)

    if not shard_dfs:
        raise SystemExit("没有可用于评估的样本。")

    combined = pd.concat(shard_dfs, ignore_index=True)
    print("\n===== 跨模型汇总（按分片大小加权均值）=====")
    for m in metrics:
        name = m.name
        if name in combined.columns:
            print(f"  {name:18s}: {combined[name].mean():.4f}")

    try:
        combined.to_csv(out_path, index=False)
        print(f"\n已保存逐条明细（含 _judge_model 列）到: {out_path}")
    except Exception as e:  # 保存失败不影响结果展示
        print(f"保存 CSV 失败（不影响结果）: {e}")


if __name__ == "__main__":
    main()

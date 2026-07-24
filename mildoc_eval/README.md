# mildoc_eval · RAG 质量评估模块

独立于 `mildoc_wxkf` 的评估工具（开发 / CI 用途，不随服务部署）。
**复用线上同一套检索 / 生成管线**，因此评估分数真实反映生产行为，而非另写一份逻辑。

## 它能做什么

对一份「黄金问答集」自动跑完整 RAG 链路，并给出量化质量指标：

| 指标 | 含义 |
|---|---|
| `faithfulness` | 答案是否忠于检索内容（有没有胡编） |
| `answer_relevancy` | 答案是否切题 |
| `context_precision` | 召回的上下文是否相关（需 reference） |
| `context_recall` | 召回的上下文是否覆盖标准答案（需 reference） |

## 运行步骤

```bash
cd mildoc_eval
pip install -r requirements.txt        # 需在与 mildoc_wxkf 相同的 Python 环境（共享 llama_index / pymilvus 等）

# 1) 准备数据：编辑 datasets/golden_set.jsonl（见 datasets/README.md）
# 2) 运行（自动读取 ../mildoc_wxkf/.env 配置）
python run_eval.py
# 或指定自定义数据集：
python run_eval.py path/to/your.jsonl
```

结果会在终端打印，并保存逐条明细到 `eval_results.csv`。

## 原理

1. `run_eval.py` 先把 `../mildoc_wxkf` 加入 `sys.path`，并 `load_dotenv` 加载 wxkf 的 `.env`
   （Milvus / Redis / LLM / Embedding 等配置全部复用，无需另配）。
2. 按 `Config.RETRIEVER_MODE` 选择 `DefaultRetrievalPipeline` / `HierarchicalRetrievalPipeline`，
   调用 `get_query_engine().query(question)` 得到答案全文 + 召回上下文。
3. 用与线上相同的百炼端点构造 Ragas 的 `llm` / `embeddings`
   （embedding 的 `chunk_size=10`，对齐百炼单次上限）。
4. 组装 `EvaluationDataset` 并 `evaluate(...)` 出分。

## 注意事项

- **embedding 百炼限制**：单次请求上限 10 条，代码已设 `chunk_size=10`，不要改大。
- **评估是离线的**：基于你准备的精选问答集，不代表生产真实用户流量；
  接入真实客流后可扩展为线上遥测评估。
- **hierarchical 模式**：若 `mildoc_wxkf/.env` 中 `RETRIEVER_MODE=hierarchical`，
  评估会自动走层次化检索（需摄取侧也用了 `hierarchical` 节点，否则召回为空）。
- 跑评估会额外消耗百炼调用（faithfulness / answer_relevancy 用 LLM-as-judge），
  数据集别一次塞太大，避免触发限流。

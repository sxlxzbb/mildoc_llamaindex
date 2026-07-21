"""LLM 生成服务（流式 / 非流式）

调用与 mildoc_index 相同的 DashScope 兼容接口生成回答。
"""
from openai import OpenAI

from config.config import Config
from logger import logger


_client = None


def get_llm_client() -> OpenAI:
    """获取 LLM 的 OpenAI 兼容客户端单例"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
        )
        logger.info("LLM 客户端初始化完成")
    return _client


# SYSTEM_PROMPT = """你是 MilDoc 智能问答助手。请仅根据【参考资料】中的内容回答用户的问题。
# 要求：
# 1. 回答要准确、简洁，使用与用户问题相同的语言（默认中文）。
# 2. 如果【参考资料】中没有相关信息，请明确说明“根据现有资料无法回答该问题”，不要编造。
# 3. 可以在回答末尾注明信息来源的文件名（file_path）。
# """

SYSTEM_PROMPT =  """
    你是一位专业的客服人员，请根据提供的参考资料内容来回答用户的问题。
    
    回答要求：
    1.【角色定位】你是一位专业、耐心、友善的客服代表
    2.【回答原则】严格基于知识库内容回答，不得编造或推测信息
    3.【准确性要求】
        - 如果知识库中有明确答案，请准确完整地回答
        - 如果知识库中信息不完整，说明现有信息并提示用户可联系人工客服获取更详细信息
        - 如果知识库中完全没有相关信息，请礼貌地说明无法找到相关资料，建议用户转接人工客服
        - 如果知识库内容中含有URL，请不要对URL做任何的改动，原样返回
    4.【回答格式】
        - 使用markdown格式
        - 语言简洁明了，适合微信对话环境
        - 使用礼貌、专业的语调
        - 如需列举，使用数字序号或简单的分行
    5.【转人工提示】当遇到以下情况时，主动建议用户转接人工客服：
        - 复杂的售后问题
        - 需要个人账户信息查询的问题
        - 投诉或纠纷相关问题
        - 知识库无法覆盖的专业技术问题
    
    请基于以上要求，为用户提供专业的客服回答。
    """


def build_messages(query: str, context_chunks: list, history: list = None) -> list:
    """组装发给 LLM 的 messages。

    Args:
        query: 当前用户问题
        context_chunks: 重排后的 RetrievedChunk 列表（作为参考资料）
        history: 历史对话 [{"role": "user"/"assistant", "content": "..."}]
    """
    context = "\n\n".join(
        f"[{i + 1}] (来源: {c.file_path or '未知'})\n{c.content}"
        for i, c in enumerate(context_chunks)
    )
    system = f"{SYSTEM_PROMPT}\n\n【参考资料】\n{context}\n"
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def stream_completion(messages: list, max_tokens: int = Config.MAX_TOKENS):
    """流式生成，yield 每个文本片段（str）"""
    client = get_llm_client()
    stream = client.chat.completions.create(
        model=Config.LLM_MODEL_NAME,
        messages=messages,
        temperature=Config.TEMPERATURE,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def complete_completion(messages: list, max_tokens: int = Config.MAX_TOKENS) -> str:
    """非流式生成，返回完整文本"""
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=Config.LLM_MODEL_NAME,
        messages=messages,
        temperature=Config.TEMPERATURE,
        max_tokens=max_tokens,
        stream=False,
    )
    return resp.choices[0].message.content or ""

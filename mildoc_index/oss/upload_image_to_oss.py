import atexit
import concurrent.futures
import os
import re
import threading
import time
import uuid

import oss2

from config.config import Config
from logger.logging import setup_logging

logger = setup_logging()

# 全局共享线程池：所有 UploadImageToOSS 实例、所有请求共用，限制总线程数，避免每次调用都新建线程池导致线程膨胀
_OSS_UPLOAD_EXECUTOR = None
_OSS_UPLOAD_EXECUTOR_LOCK = threading.Lock()
# I/O 密集型任务，默认线程数为 CPU 核数 * 2；可通过 OSS_UPLOAD_MAX_WORKERS 显式覆盖
_OSS_UPLOAD_MAX_WORKERS = Config.OSS_UPLOAD_MAX_WORKERS


def _get_oss_upload_executor() -> concurrent.futures.ThreadPoolExecutor:
    """获取（懒加载）全局共享线程池，进程退出时自动关闭。"""
    global _OSS_UPLOAD_EXECUTOR
    if _OSS_UPLOAD_EXECUTOR is None:
        with _OSS_UPLOAD_EXECUTOR_LOCK:
            if _OSS_UPLOAD_EXECUTOR is None:
                _OSS_UPLOAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_OSS_UPLOAD_MAX_WORKERS,
                    thread_name_prefix="oss-upload",
                )
                atexit.register(_shutdown_oss_upload_executor)
    return _OSS_UPLOAD_EXECUTOR


def _shutdown_oss_upload_executor():
    global _OSS_UPLOAD_EXECUTOR
    if _OSS_UPLOAD_EXECUTOR is not None:
        _OSS_UPLOAD_EXECUTOR.shutdown(wait=True)
        _OSS_UPLOAD_EXECUTOR = None

class UploadImageToOSS:
    def __init__(self):
        self.access_key_id = Config.OSS_ACCESS_KEY_ID
        self.access_key_secret = Config.OSS_ACCESS_KEY_SECRET
        self.endpoint = Config.OSS_ENDPOINT
        self.bucket_name = Config.OSS_BUCKET_NAME
        self.remote_path = Config.OSS_IMAGE_PATH
        self._init_oss()

    def _init_oss(self):
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    def _upload_single_image_sync(self, args):
        """
        同步上传单张图片（用于线程池）
        :param args:
        :return:
        """
        local_path, remote_path = args

        try:
            # 检查文件是否存在
            if not os.path.exists(local_path):
                logger.logging(f"图片不存在:{local_path}")
                return None

            # 上传
            with open(local_path, 'rb') as f:
                self.bucket.put_object(remote_path, f)

            # 生成URL，桶是私有的，所有需要生成签名URL
            # url = self.bucket.sign_url('GET', remote_path, 3600) # 1小时有效
            # 如果桶是公有的
            url = f"https://{self.bucket_name}.{self.endpoint}/{remote_path}"
            # 如果有cdn加速
            # url = f"https://{self.cdn_domain}/{remote_path}"

            return url, local_path
        except Exception as e:
            logger.error(f"上传OSS异常，local_path:{local_path}, remote_path:{remote_path}", e)


    def upload_markdown_image_to_oss(self, md_content: str, file_path: str) -> str | None:
        """
        使用线程池处理Markdown中的图片（将图片识别出来，然后上传到oss）
        :param md_content: markdown文件内容
        :param file_path: 文件路径(各种类型的文件)，图片在该文件相同路径的iamges文件夹下
        :return:
        """

        start_time = int(time.time() * 1000)

        file_name = os.path.basename(file_path)

        if not md_content:
            logger.info(f"图片上传oss,入参Markdown文件内容为空:{file_name}")
            return None

        # 文档目录
        path_name = os.path.dirname(file_path)
        # 图片目录
        image_path = os.path.join(path_name, Config.DOC_IMAGE_DIR)
        if not image_path:
            logger.info(f"图片路径不存在,file_path:{file_path}, image_path:{image_path}")
            return md_content

        # 提取图片
        pattern = r'!\[(.*?)\]\((.*?)\)'
        images = re.findall(pattern, md_content)

        if not images:
            logger.info(f"图片上传oss,该markdown文件没有引用图片：{file_name}")
            return md_content

        logger.info(f"{os.path.basename(file_name)}找到{len(images)}张图片")

        # 准备上传
        image_oss_path = self.remote_path or str(uuid.uuid4())
        tasks = []
        # 这儿alt_text看作是文件类型(暂时没用)，local_path是图片在本地的存放路径
        for alt_text, local_path in images:
            # 这儿重新拼接一遍图片路径是为了防止文档里使用相对路径
            image_base_name = os.path.basename(local_path)
            full_path = os.path.join(image_path, image_base_name)
            if not os.path.exists(full_path):
                logger.error(f"本地不存在图片:{full_path}")
                continue

            remote_path = f"{image_oss_path}/{image_base_name}"

            tasks.append((full_path, remote_path))

        # 使用全局共享线程池并发上传（避免每次调用都新建线程池导致线程膨胀）
        url_map = {}
        executor = _get_oss_upload_executor()
        # 提交所有任务
        future_to_task = {executor.submit(self._upload_single_image_sync, task): task for task in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            if not result:
                continue

            url, local_path2 = result

            for alt_text, local_path in images:
                full_path = os.path.join(image_path, os.path.basename(local_path))
                if full_path == local_path2:
                    url_map[local_path] = url
                    break

        # 替换内容
        def replace_fn(match):
            alt_text1 = match.group(1)
            local_path1 = match.group(2)
            if local_path1 in url_map:
                return f"![{alt_text1}]({url_map[local_path1]})"
            return match.group(0)

        new_md_content = re.sub(pattern, replace_fn, md_content)

        logger.info(f"图片替换完成,{file_name}, 耗时:{int(time.time() * 1000) - start_time}ms")

        return new_md_content


if __name__ == '__main__':
    upload = UploadImageToOSS()
    md_file_path = r'D:\zbb\test\temp\cf47256c-323c-45d5-8f61-36cbc1a26647\MongoDB-test.md'
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = upload.upload_markdown_image_to_oss(content, md_file_path)

    output_path = md_file_path.replace('.md', '_online.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"📄 ✅ 处理完成！输出文件: {output_path}")


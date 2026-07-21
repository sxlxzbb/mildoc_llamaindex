CREATE TABLE `users` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `username` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户名',
    `password` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '密码',
    `nickname` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '' COMMENT '昵称',
    `status` tinyint NOT NULL DEFAULT '1' COMMENT '状态：1-正常，0-删除',
    `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

CREATE TABLE `chat_history` (
    `id`         BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `username`   VARCHAR(50)  NOT NULL                COMMENT '用户名',
    `session_id` VARCHAR(64)  NOT NULL DEFAULT ''     COMMENT '会话ID（一次完整对话）',
    `role`       VARCHAR(20)  NOT NULL                COMMENT '角色：user/assistant/system',
    `content`    TEXT         NOT NULL                COMMENT '消息内容',
    `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_username` (`username`),
    KEY `idx_session` (`session_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='长期对话记忆归档';

# 数据来源与处理

原始社区数据来自 https://huggingface.co/datasets/ams-99/fakeddit_9k 。作者官方元数据通过 https://github.com/entitize/Fakeddit 链接取得。原文件校验记录为 source-files.json、official-source-files.json。

8646 条社区数据中，8562 条通过规范化标题与原始图片 URL 唯一匹配官方 ID，且标签、划分均一致；84 条无法唯一对应。来源核验后的清单排除这些记录。逐条与汇总记录为 provenance-matches.jsonl、provenance-audit.json。

第一阶段固定清单：308/65/64 条，精确文本/解码图像去重。第二阶段来源核验前与核验后均为 1007/305/304 条，加入 dHash 距离 ≤3 候选过滤。对应清单在 data/processed/phase1、phase2-exploratory、phase2-verified。本次 PyTorch 重跑保持清单 SHA256 不变，未重选测试集。

类别 4 稀缺，因此数据不能称作平衡数据，严格 8-shot 无法运行。Fakeddit 标签使用基于来源的远程监督，不表示每条样本都经人工事实鉴定；详见 https://arxiv.org/html/1911.03854 。尚需人工近重复检查与域外评估，当前不重新分发数据。

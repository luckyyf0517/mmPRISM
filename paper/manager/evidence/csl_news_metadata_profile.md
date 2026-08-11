# CSL-News Metadata Profile Evidence

Status: `partial_evidence_ready`
Last Updated: `2026-08-11`
Role: `EVID-REV-DATASET_public_metadata_component`

## 1. Provenance

```text
source: huggingface:ZechengLi19/CSL-News
revision: 3a0601210333fe760efd09b5d9e2ae5f341ce339
profile schema: mmprism.csl_news_metadata_profile.v1
artifact: /mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_metadata_profile_v1/profile_20260811T151215Z.json
artifact SHA-256: 90e24aa4236febca9aa5bc8faaa025751618210f05d6bfd32d76ab9d94f10c43
status: passed_with_warnings
```

复现命令：

```bash
scripts/run_csl_news_metadata_profile.sh
```

报告同时绑定 labels JSON、labels CSV 和 dataset card 的路径、字节数和 SHA-256；输出使用原子写入。

## 2. 已验证事实

| Item | Result |
|---|---:|
| 官方声明的手语类型 | Chinese Sign Language |
| 官方 language code / task / license | `zh` / `video-text-to-text` / CC BY-NC 4.0 |
| JSON 视频-译文标注单元 | 722,711 |
| 有效、非空 JSON 记录 | 722,711 |
| 唯一 video / pose key | 722,711 / 722,711 |
| video-pose stem mismatch | 0 |
| 唯一规范化译文 | 686,839 |
| 重复译文记录 | 35,872 |
| 平均 lexical codepoints/译文 | 34.8551 |
| lexical codepoints 中位数 / p95 / max | 33 / 66 / 112 |
| 平均 Han codepoints/译文 | 34.8269 |
| 译文 Han character set size | 4,805 |
| 来源文件名分类 | Common-Concerns 488,425；Dragon-TV 227,138；unknown 7,148 |

这里的 `lexical codepoints` 排除 Unicode punctuation/separator/control 类别。`4,805` 仅是自然语言
译文中的汉字字符集大小，不是 sign/gloss vocabulary，不得在正文中改写成手语词汇量。

## 3. JSON/CSV 差异

JSON 是 canonical annotation source：722,711 个 video key 全部唯一。CSV 有 722,715 行和相同的
722,711 个唯一 video key；所有 canonical JSON 记录都能在 CSV 中找到，但 CSV 对以下 4 个 key
各多出一条冲突译文：

- `20231116_Dragon-TV__9837-9912_17296.mp4`
- `Common-Concerns_20200209_51012-51087_94740.mp4`
- `Common-Concerns_20210723_29262-29337_75638.mp4`
- `Dragon-TV_20230305_58012-58080_167503.mp4`

处理规则：不修改上游文件；canonical pipeline 只读取唯一 JSON；CSV 仅作交叉审计，不允许
last-write-wins 或静默覆盖。首次发现差异的 failed profile 也保留在同一 artifact 目录。

## 4. Reviewer Coverage And Limits

| Reviewer 要求 | 当前证据 | 状态 |
|---|---|---|
| sign language type | dataset card 明确声明 Chinese Sign Language | ready_for_source_description |
| vocabulary size | 只有译文字符统计；没有 sign/gloss 字段 | missing_sign_vocabulary |
| sentence count | 有 722,711 个视频-译文单元；没有显式 sentence ID | segment_count_ready_sentence_count_undefined |
| average sentence length | 可报告译文 codepoint 长度，必须给出单位定义 | ready_with_scope_label |
| non-manual grammatical features | metadata 无相应字段 | missing |
| subjects/signers, scene, orientation, occlusion, split | metadata 无相应字段 | missing |

因此 `ED-SCI-4`、`R1-4b`、`R2-4` 只能从 `blocked` 推进到 `in_progress`，不能标记完成。后续还需：

1. 确认原稿实际使用的是完整 CSL-News 还是某个子集，并生成对应 manifest hash；
2. 从视频/原始数据恢复 signer、场景、方向、遮挡和 non-manual 可用性，无法恢复时在限制中声明；
3. 对真实 12 人和新增 stress set 分别给出 subject/session/split 统计与 leakage audit；
4. 将最终数字写入主文/补充材料前，再从 frozen manifest 重新生成 profile。

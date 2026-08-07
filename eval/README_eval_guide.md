# 泰晶 OCXO/晶振 售前问答 评测集 — 工程指南 (v0.1)

面向新手的、可落地的评测集建设说明。第一阶段聚焦 **端到端答案质量**
（系统读证据生成的中文/英文答案好不好），语料 = `corpora/tkd/data`（泰晶 74 份
datasheet，547 个版面 chunk）。

---

## 0. 我们在评什么

你的链路有三段，每段都会出错：

```
用户需求 ──①可选查询改写──> ②双路召回(dense+BM25+RRF, top_k=15) ──> ③可选多模态模型读证据并输出带引用答案
```

第一阶段我们评 **③ 的产出 = 最终答案**，但评测样本同时标注 **②应该召回的
证据块**，这样一旦答案错，能区分是「没召回对」还是「召回对了但答错」。

### 关键事实：泰晶是扫描件，chunk 文本是空的

`native_text` 基本为空，chunk 的语义来自 Gemini 生成的英文 `description`，
**而 description 是有损的**——有的带数值，有的只列出"有这个参数"。真正精确的
数值（如相位噪声 -150 dBc/Hz）只在 **裁剪图** 里，需要由已配置的多模态模型在答题时读图得到。

> **由此得出标注铁律：gold_facts（标准事实）必须以"看原始 PDF/裁剪图"为准，
> 不能只抄 chunk 的 description。** 否则标准答案本身就是错的。

---

## 1. 一条评测样本的结构 (schema)

每条样本是一个 JSON 对象（评测集存成 JSONL，一行一条）。字段：

| 字段 | 必填 | 含义 |
|------|------|------|
| `id` | ✓ | 唯一编号，如 `tkd-0007` |
| `question` | ✓ | 客户的真实问法（中文或英文） |
| `lang` | ✓ | `zh` / `en` |
| `persona` | ✓ | 提问角色：`硬件工程师`/`采购`/`系统架构师`/`代理FAE` |
| `intent` | ✓ | 意图类型，见 §2 |
| `difficulty` | ✓ | `easy`/`medium`/`hard` |
| `answerable` | ✓ | 语料能否回答。`false` = 库外问题（考幻觉/拒答） |
| `products` | | 涉及型号，如 `["TOC2522","TOC2525"]` |
| `gold_chunks` | ✓ | 正确证据块的 `chunk_id` 列表（来自 `corpora/tkd/data/index/chunks.jsonl`） |
| `gold_facts` | ✓ | 答案**必须包含且正确**的关键事实（数值带单位） |
| `must_not` | | 绝不能出现的内容（编造值、错单位、瞎给报价等） |
| `verify_status` | ✓ | gold 的可信度来源，见下 |
| `notes` | | 标注备注（陷阱点、来源页等） |

`verify_status` 取值：
- `verified_image` — 我已读原始 PDF/裁剪图核对过数值（最高可信）
- `from_description` — 取自 chunk description，**待图核**
- `unverified` — 仅结构正确，gold 还需人工补全

> 新手重点：`gold_chunks` + `gold_facts` 是评测集的灵魂。没有它们就只能"人眼看
> 感觉"，无法自动算分、无法版本回归。

---

## 2. 意图分类 (intent taxonomy) 与配额

按真实售前场景分布，建议第一批 ~40 条按下表配额（括号为建议条数）：

| intent | 说明 | 占比 |
|--------|------|------|
| `spec_lookup` 单参数查询 | "TOC2522 老化率多少" | 8 |
| `multi_param` 多参数/整表 | "把电性能参数都列出来" | 4 |
| `compare` 多型号对比 | "2522 和 2525 相位噪声差别" | 6 |
| `application_fit` 应用选型 | "5G基站时钟用哪颗" | 6 |
| `constraint_filter` 带约束筛选 | "工作温度要到 -40℃ 的" | 5 |
| `option_lookup` 可选项 | "有哪些频点/稳定度档位" | 3 |
| `cross_reference` 竞品交叉 | "对标 Rakon 某型号有等效件吗" | 3（多为库外） |
| `commercial` 商务 | "MOQ/交期/价格" | 3（**库外，考拒答**） |
| `package_mech` 封装/机械 | "尺寸/焊盘/管脚定义" | 2 |

**必须包含的压力测试题**（最能暴露问题，新手最易漏）：
1. **库外商务题**（`answerable=false`）→ 正解是"资料无此信息，建议联系泰晶"，不得编造。
2. **单位陷阱**：10年稳定度是 **ppm** 而非 ppb（同表其它行是 ppb）。
3. **偏移点陷阱**：相位噪声 @10Hz vs @10kHz 取错。
4. **min/typ/max 取错**、**约束/否定**（"不要正弦波的"）。
5. **多跳/跨表/跨型号**。

---

## 3. 端到端答案质量 评分维度

对每条样本，给系统答案打分（建议 LLM-as-judge + 人工抽检，judge 必须**看到
gold_facts 和证据**，即"有参考答案的判分"，不是盲判）：

| 维度 | 判定 | 说明 |
|------|------|------|
| **事实覆盖** Coverage | gold_facts 命中比例 | 该说的关键事实说全了吗 |
| **数值正确** Numeric | pass/fail | 数字+单位完全一致（最严，工程文档命根子） |
| **忠实度** Faithfulness | pass/partial/fail | 每个论断都能在证据里找到支撑，无幻觉 |
| **引用正确** Citation | pass/partial/fail | 标的 [E#] 真支撑那句话；关键论断有引用 |
| **拒答正确** Refusal | pass/fail | 仅对 `answerable=false`：正确说"无此信息/联系工厂" |
| **must_not 违规** | 有/无 | 命中任一 must_not 即整条判 fail |

聚合指标：各维度通过率；`answerable=false` 子集单独看拒答率（这是幻觉护栏的
核心 KPI）。

---

## 4. 工作流（我们正在走的）

1. **吃透语料** ✓（已完成：74 文档 / 547 chunk / 产品族与 block_type 已盘点）
2. **定 schema + taxonomy** ✓（本文件）
3. **造种子集**（本批：`seed_eval.jsonl`，含已核对样例）← 现在
4. **你复核**：问法是否拟真？gold 是否正确？配额是否合理？
5. **扩量到 ~40**：按配额补齐，逐条回图核对 gold（把 `from_description` 升级为
   `verified_image`）
6. **试跑 + 校准**：接你的 `/api/answer` 跑一遍，用 §3 维度判分，看指标是否区分得开
7. **迭代 + 版本化**：评测集进 git，每次改动打 tag，便于回归对比

---

## 5. 新手三大坑（务必避免）

- **照 datasheet 措辞抄题** → 不像真实客户问法。要用客户口吻（"我在做X，需要Y"）。
- **让同一个 LLM 出题+出答案+又判分** → 自说自话。出题可借助 LLM，但 gold 必须
  回到原始证据核对；judge 与被测模型尽量不同源。
- **只有顺风题** → 漏掉库外/陷阱题，测不出幻觉。压力测试题必须占一定比例。

---

## 文件清单

- `README_eval_guide.md` — 本指南
- `schema.json` — 样本字段的机器可读 schema
- `seed_eval.jsonl` — 种子评测集（首批样例，含 grounded gold）
- 后续：`eval_run.py`（跑分脚本）、`results/`（每次评测结果与版本）

# SemEval 2027 Task5

英语名词复合词（EN NN）语境组合性回归实验。训练2776句、内部验证704句，词组及共享句子不跨折；单训练种子2027，没有使用官方Dev／Test选参。

- [第一周实验记录](实验记录.md)：数据、方法和结果。

**当前主指标最高：Qwen3.5-9B软标签CE，平均Spearman 0.6733；平均分量RMSE最低：Qwen3.5-4B，0.5837。** 都是单种子Internal Val，尚未确认稳定差异。

## 关键代码入口

| 内容 | 文件 |
|---|---|
| 数据分组划分、Spearman与RMSE | [src/task5_data.py](src/task5_data.py) |
| DeBERTa双回归头 | [src/train_encoder.py](src/train_encoder.py) |
| Qwen末位表示／平均池化 | [src/train_qwen.py](src/train_qwen.py)、[src/train_qwen_mean.py](src/train_qwen_mean.py) |
| 数值期望、软标签CE、RLOO与REAL损失 | [src/regression_methods/core.py](src/regression_methods/core.py) |
| 首轮CE／RAFT／TRACT／REAL训练与验证 | [src/regression_methods/train.py](src/regression_methods/train.py) |
| Qwen3.5原生LM head及CE对照 | [src/ce_model_compare_v3.py](src/ce_model_compare_v3.py) |
| 后续原生思考与评分边界 | [src/thinking_protocol.py](src/thinking_protocol.py)、[src/thinking_model.py](src/thinking_model.py)、[src/train_thinking.py](src/train_thinking.py) |
| 32句推理诊断 | [scripts/run_native_inference_diagnostic.py](scripts/run_native_inference_diagnostic.py) |
| 已确认缺陷的CPU探针 | [tests/paper_audit_contract.py](tests/paper_audit_contract.py) |

`src/`保留实际训练／诊断代码，不为发布重写算法。旧TRACT的内部伪评分缺陷仍在历史代码中，已明确标注，**不建议原样重跑旧解释路线作为新结果**。原计划A1／A2思考训练和后续B组尚未完成；写有损失函数不等于已有质量结果。

`experiments/`是原始冻结配置，包含计划未执行部分；完成状态以[结果汇总](results/first_week_results.json)为准。`snapshots/a0_v4/src/`保存A0训练时与当前`src/`不同的三个文件。`publication_manifest.json`记录发布文件SHA、run来源和纸本PDF哈希。

## 验证与使用范围

仅标准库检查：

```bash
python3 -m unittest discover -s tests -p 'test_data_metrics.py' -v
python3 tests/thinking_protocol_contract.py
python3 tests/native_inference_diagnostic_contract.py
```

在相应PyTorch运行环境下检查（无需真实4B／9B权重）：

```bash
# Qwen3旧环境，或兼容其接口的环境
python tests/methods_contract.py
python tests/paper_audit_contract.py

# Qwen3.5环境
PYTHONPATH=src python tests/ce_models_contract_v3.py
PYTHONPATH=src python tests/thinking_model_contract.py
```

实际运行环境版本见`environments/`。CPU tiny模型测试通过不代表真实模型精度；新探针的PASS包含“旧缺陷已复现”，不代表缺陷已修复。

这是关键代码及结果快照。真实训练仍需另备官方数据、本地底座权重及`DOWNLOAD_VERIFIED.json`、冻结Train／Val文件与`split.json`；训练入口还保留原服务器GPU UUID和空卡检查。迁移机器时应另建配置、重新校验；不直接执行全部冻结队列。原生思考完整诊断还依赖未上传的大体积checkpoint与历史CPU证明，不能将本仓库描述为开箱即跑的全量复现实验包。

仓库不包含模型、训练语料、逐样本预测、教师解释缓存或论文PDF。完整结果可追溯，但当前只适用于英语NN、单内部划分；官方scorer一致性、多训练种子和修复后的质量收益仍待验证。

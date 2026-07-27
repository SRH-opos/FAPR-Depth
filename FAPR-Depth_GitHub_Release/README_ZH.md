# FAPR-Depth 开源仓库说明

该压缩包按照常见论文代码仓库结构整理，包含最终训练、测试、消融、
可视化和论文分析脚本。数据集、第三方Backbone、缓存文件和模型权重
没有放入压缩包，需要在公开前自行补充。

## 公开前最重要的事项

1. 修改README与CITATION.cff中的GitHub用户名。
2. 与所有作者确认MIT许可证是否合适。
3. 删除所有私人路径、邮箱、Token和未公开数据。
4. 检查第三方Backbone的许可证，不能直接无授权复制。
5. 上传最终权重并给出SHA-256校验值。
6. 使用一台干净环境完成训练、测试和单样本推理验证。

## 推荐的GitHub首页目录

- `assets`：架构图、定性结果和说明图片
- `configs`：实验配置
- `datasets`：数据准备说明与数据接口
- `models`：模型说明
- `analysis`：论文机制分析脚本
- `experiments`：消融、SOTA比较和效率实验
- `scripts`：运行脚本
- `third_party`：第三方依赖说明
- `weights`：权重下载说明
- 根目录：`train.py`、`test.py`、`inference.py`、`README.md`

# DeepEyes：通过强化学习激励"用图像思考"

这个目录包含在verl框架内复现DeepEyes论文的实现，支持多轮视觉工具调用。这个实现基于原始的[DeepEyes论文](https://arxiv.org/abs/2505.14362)及其[官方实现](https://github.com/Visual-Agent/DeepEyes)，集成了verl框架的多模态和多轮能力。

## 复现实验

> **关于'Chart'数据集的说明：**
>
> 提供的预处理脚本有意排除了`data_v0.8_visual_toolbox_v2.parquet`，其中包含'Chart'数据。这个子集由非常高分辨率的图像组成，通常类似于由多个子图组成的大型图形，很像学术论文中的图形。
>
> 因此，即使使用放大工具，裁剪后的图像仍然很大。这造成了显著的内存不足（OOM）错误风险，可能导致训练过程突然终止。
>
> **我们强烈建议不要在单节点上训练'Chart'数据集。**

> **关于'thinklite'数据集的说明：**
> `thinklite`数据集中的许多图像分辨率非常低，高度或宽度都低于28像素。这达不到Qwen-2.5VL图像处理器所需的最小输入尺寸，会在数据加载期间导致错误。
>
> 为了缓解这个问题，我们将这些低分辨率图像放大以满足处理器的要求。但是请注意，由于原始分辨率较低，后续放大工具的`crop`操作可能经常触发异常，这反过来可能影响模型的工具使用性能。

首先，启动推理服务作为奖励计算的评判器。您可以使用以下脚本作为参考：

```bash
python -m sglang.launch_server --model-path /path/to/Qwen2.5-72B-Instruct \
    --port 18901 \
    --tp-size 8 \
    --context-length 32768 \
    --trust-remote-code \
    --log-requests false
```

接下来，您可以开始训练：

```bash
bash recipe/deepeyes/run_deepeyes_grpo.sh
```

## 性能

![score](https://private-user-images.githubusercontent.com/82520804/474784419-b13f4f72-bb3a-4281-a43b-1f34a9037c0c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ0NTQxMTMsIm5iZiI6MTc1NDQ1MzgxMywicGF0aCI6Ii84MjUyMDgwNC80NzQ3ODQ0MTktYjEzZjRmNzItYmIzYS00MjgxLWE0M2ItMWYzNGE5MDM3YzBjLnBuZz9YLUFtei1BbGdvcml0aW09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODA2VDA0MTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJjNGMxMjhiOGM4MTNhYTEzYTE2MTYzY2ZjYWRhNmEzMmVjNjUxOGI3MTgzOGQyM2ZmOWJlYTZlNDYzYzU0ZDkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.qTDX-3fyLHWdeFh9o4b6nIAB57bT0XyLjKXhNV6k5nA)

![entropy](https://private-user-images.githubusercontent.com/82520804/474785253-752106a9-e25d-4b44-aef9-1ac98015d05c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ0NTQxMTMsIm5iZiI6MTc1NDQ1MzgxMywicGF0aCI6Ii84MjUyMDgwNC80NzQ3ODUyNTMtNzUyMTA2YTktZTI1ZC00YjQ0LWFlZjktMWFjOTgwMTVkMDVjLnBuZz9YLUFtei1BbGdvcml0aW09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODA2VDA0MTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM4OGQ2ZGI3M2JlYWE4YTQyMzIxMWYxMzZhNDBmNmYxNzcwNDgxNThiZDRiMzQyYzUwZjc3OWE4YzdhYWEwMWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.PhimMTxXXEtMLPGzejPQuw-Ul0As8ey-hyy1qkeABIQ)

![num_turns](https://private-user-images.githubusercontent.com/82520804/474785462-c99c7952-14db-485a-acd2-14e5956ecc34.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ0NTQxMTMsIm5iZiI6MTc1NDQ1MzgxMywicGF0aCI6Ii84MjUyMDgwNC80NzQ3ODU0NjItYzk5Yzc5NTItMTRkYi00ODVhLWFjZDItMTRlNTk1NmVjYzM0LnBuZz9YLUFtei1BbGdvcml0aW09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODA2VDA0MTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJkNWYwMGVjOWM4NDVhZTkzZWI5NWMzMGVjZTcyZGM2NDExY2FmYTBlYWJmZTk5YTU5MzM3NmNkYWI4Y2U4Y2YmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Ieakk_ttMsNygVzpZZqGs1507j2GC-rqHSYH9iQQ71Q)

更多详情请参见[评论](https://github.com/volcengine/verl/pull/2398#issuecomment-3157142856)。

注意：AgentLoop不直接记录num_tool_calls，而是记录num_turns。在我们的场景中，您可以通过num_tool_calls = num_turns / 2 - 1来计算工具调用次数。

## 参考和致谢

- [DeepEyes论文](https://arxiv.org/abs/2505.14362)
- [DeepEyes官方实现](https://github.com/Visual-Agent/DeepEyes)

---
如果您需要更多复现详情或遇到任何问题，请随时提出issue或联系维护者。
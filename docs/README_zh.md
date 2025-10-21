# verl 文档

## 构建文档

```bash
# 如果您想查看自动生成的 API 文档字符串，请确保 verl 在 Python 路径中可用。例如，通过以下方式安装 verl：
# pip install .. -e[test]

# 安装构建文档所需的依赖。
pip install -r requirements-docs.txt

# 构建文档。
make clean
make html
```

## 使用浏览器打开文档

```bash
python -m http.server -d _build/html/
```
启动浏览器并导航到 http://localhost:8000 来查看文档。或者，您可以将文件 `_build/html/index.html` 拖到本地浏览器中直接查看。
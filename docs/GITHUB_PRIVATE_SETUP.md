# 上传到 GitHub 私有仓库

1. 登录 GitHub，新建仓库，Visibility 选择 `Private`。
2. 在本文件夹打开 PowerShell。
3. 执行：

```powershell
git init
git add .
git commit -m "initial private four-strategy package"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

4. 邀请同学：GitHub 仓库页 `Settings -> Collaborators -> Add people`。
5. 如果以后必须上传超过 100MB 的中间表，再启用 Git LFS；当前包已规避超大文件。

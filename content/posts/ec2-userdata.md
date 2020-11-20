---
title: "EC2のデフォルトユーザー(ec2-user)とデフォルトSSHポートの変更をユーザーデータで自動化する"
description: EC2のデフォルトユーザー(ec2-user)とデフォルトSSHポートの変更をユーザーデータで自動化する方法を確認します。
date: 2020-11-20T15:00:27+09:00
draft: false
---

EC2の起動時のオプションにユーザーデータという項目があります。
こちらを利用すると立ち上げ時のOS内部の設定を変更できます。
例えばEC2ではデフォルトでec2-userという名称のユーザーが作られますが、このデフォルトユーザーを任意の名称に変更することも可能です。

今回は例として「デフォルトユーザーの名称変更」と「SSHポートの変更」をやってみます。

と言ってもやることは簡単で、EC2起動時の項目に「ユーザーデータ」というのがあるのでそこに以下のyaml形式のテキストを入力して立ち上げるだけです。

以下の `<username>` は任意のユーザー名で置き換えます。


```yaml
#cloud-config
timezone: Asia/Tokyo
users:
  - default
runcmd:
  - '/bin/sed -i".org" --follow-symlinks -e "s/#Port 22/Port 10022/" /etc/ssh/sshd_config'
  - 'systemctl restart sshd.service'
system_info:
  default_user:
    name: <username>
```

以下のコマンドでSSHログインします。

```sh
ssh -i <path-to-key> <username>@<hostname> -p 10022
```
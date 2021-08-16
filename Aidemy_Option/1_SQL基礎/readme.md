# サマリ
- 本当に基礎の基礎だった
- 一通り再確認

# メモ
## データを選択―SELECT文
```
SELECT <カラム名> FROM <テーブル名>
```

## データの条件付け―WHERE句
```
SELECT <カラム名> FROM <テーブル名> WHERE <条件式>
```

## 条件を組み合わせ―AND句、OR句
```
SELECT <カラム名> FROM <テーブル名> WHERE <条件式1> AND <条件式2>
SELECT <カラム名> FROM <テーブル名> WHERE <条件式1> OR <条件式2>
```

## 並び替える―ORDER BY句
```
SELECT <カラム名> FROM <テーブル名> ORDER BY <カラム名> <ASC または DESC>
```

## 制限してズラす―LIMIT句、OFFSET句
- LIMIT は取得数の制限
- OFFSET は取得開始をずらす
```
ORDER BY <カラム名> <ASC or DESC> LIMIT <制限数> OFFSET <取得開始数>
```

## レコードの追加―INSERT文
```
INSERT INTO <テーブル名> (<カラム1>, <カラム2>, <カラム3>,...) VALUES (<値1>, <値2>, <値3>,...)
```

## データを上書き―UPDATE文
```
UPDATE <テーブル名> SET <カラム1> = <値1>, <カラム2> = <値2>, ... WHERE <条件式>
```

## レコードを削除―DELETE文
```
DELETE FROM <テーブル名> WHERE <条件式>
```

## まずはここから―CREATE文
```
CREATE TABLE <テーブル名> (<カラム1> <データ型1>, <カラム2> <データ型2>, ...)
```

## 列を追加―ALTER文
```
ALTER TABLE <テーブル名> ADD <カラム名> <データ型>
```

## 名前の変更―RENAME文
```
RENAME TABLE <テーブル名> TO <新テーブル名>
```

## 扱い注意―DROP文
```
DROP TABLE テーブル名
```
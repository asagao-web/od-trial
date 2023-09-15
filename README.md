# od-trial
for dev team
検証順は
yolo v5
custom SSD
fast r cnn with resnet 50
normal SSD
SSD lite (SSD with mobilenet)
fast r cnn with mobilenet
の順です

SSD liteとfast r cnn with mobilenetが群を抜いて早かった(実行速度0.1~0.19程度)
なので、バックボーンネットワークにmobilenetを使用して、転移学習やファインチューン後に枝払いなどでモデル圧縮などをすると軽量化できるのではないでしょうか？
今はssd-liteをcocoデータセットをvoc形式で保存したデータセットで訓練する際のレポジトリを制作しています

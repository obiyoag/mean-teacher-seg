使用20cases，数据集中最后2674条是有标签的。
220cases共有29772条数据,最后20例2674条有标签  
尝试：  
0.741
半监督的改进方向：
+ 使用自监督预训练的模型
+ 数据增强
+ 改一下cli可以开多个终端训练
+ 改变一个batch里面labeled和unlabeled数据的比例
+ 使用余弦退火更新学习率

python main.py --augmentation True --best-checkpoint-name best3.ckpt

python main.py --cosine-annealing True --best-checkpoint-name best2.ckpt  0.483
python main.py --cosine-annealing True --augmentation True --best-checkpoint-name best3.ckpt      0.626
python main.py --cosine-annealing True --augmentation True --self-supervised True --best-checkpoint-name best4.ckpt    0.755


python main.py --augmentation True --best-checkpoint-name best3.ckpt   0.763
python main.py --augmentation True --self-supervised True --best-checkpoint-name best4.ckpt   0.746


0.60

基於https://github.com/regob/vehicle_reid 修改的
下載預訓練模型 resnet50_mixstyle，然後用AI_CUP資料集去訓練
https://drive.google.com/file/d/1STbsacssLtlHpUesNzuTeUPrfMlWbSKu/view


訓練reid:
cd vehicle_reid
pip install -r requirements.txt
訓練的AI_CUP競賽的ReID模型
python train.py --data_dir ../datasets  --name resnet50_ibn11 --color_jitter  --mixstyle  --train_csv_path ../datasets/AI_CUP/train_labels.csv  --val_csv_path ../datasets/AI_CUP/val_labels.csv  --batchsize 32 --total_epoch 120 --warm_epoch 3  --save_freq 5 --erasing_p 0.5 --model resnet_ibn --model_subtype 50  --contrast   --num_workers 0 --samples_per_class=4 --cosine --checkpoint C:\Users\LPCAS\Desktop\vehicle_reid\vehicle_reid\model\resnet50_mixstyle\net_19.pth --start_epoch 20 --lr=0.008




如果想使用訓練完的模型看query可視化效果及查看rank,map，已分割好test_csv,query_csv,train_csv，只需將訓練Baseline fast_reid時生成的bounding_box_test,bounding_box_train資料夾放入\vehicle_reid\datasets\AI_CUP下
打開.ipynb_checkpoints/run_ok.ipynb
照順序執行即可，如果要用不同ID車輛進行query，[19]在random.seed(666)中的666替換任意數字就可了 。


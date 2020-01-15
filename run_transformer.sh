python main.py \
--test=0 \
--save_model=1 \
--use_saved_model=0 \
--saved_path='results/model_k1.pkl' \
--saved_plot_path='results/plot_k1.png' \
--saved_pred_path='results/pred_k1.txt' \
--kernel_size=1

python main.py \
--test=1 \
--save_model=0 \
--use_saved_model=1 \
--saved_path='results/model_k1.pkl' \
--saved_plot_path='results/plot_k1.png' \
--saved_pred_path='results/pred_k1.txt' \
--kernel_size=1

python main.py \
--test=0 \
--save_model=1 \
--use_saved_model=0 \
--saved_path='results/model_k3.pkl' \
--saved_plot_path='results/plot_k3.png' \
--saved_pred_path='results/pred_k3.txt' \
--kernel_size=3

python main.py \
--test=1 \
--save_model=0 \
--use_saved_model=1 \
--saved_path='results/model_k3.pkl' \
--saved_plot_path='results/plot_k3.png' \
--saved_pred_path='results/pred_k3.txt' \
--kernel_size=3





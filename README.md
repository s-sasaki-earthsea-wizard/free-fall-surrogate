# Free-fall-surrogate

## 概要
自由落下運動をサロゲートモデルで再現することを試みます。
- サロゲートモデルは、通常は数値解析やシミュレーションでしか解けないような複雑な物理現象を、機械学習モデルを使って簡単かつ高速に再現する手法です。
- このプロジェクトでは、サロゲートモデルを用いて自由落下運動を予測し、物理ベースのシミュレーションを効率化することを目指しています。
- 初速度と発射角度という2つのパラメータから自由落下の軌道を計算し、その精度や応用可能性を評価します。

## 開発環境
- OS: Ubuntu-20.04 (Docker container)
- Python: 3.10.10

## インストール方法
Dockerをインストールした後、以下のコマンドで`./Dockerfile`を参照してイメージのビルドを行います。
```
make docker-build
```
コンテナの起動は以下のコマンドで行います。
```
make docker-run
```
起動したコンテナには必要な依存関係などはすべてインストールされた状態になっています。

## 使い方
コンテナ内で実行できるコマンドとその実行内容を説明します。

放物線データを生成する
```
make generate_parabolic_data
```
生成されたデータは
- `./data/simulation/parabolic_motion.csv`に軌道(時刻、x座標、y座標)
- `data/simulation/parabolic_params.csv`に初速度と発射角度のパラメータ
がそれぞれ保存されます。

生成した放物線データをtrain, test, valの3つに分割する
```
make split
```
分割されたデータは
- training data: `./data/simulation/splits/train_motion_data.csv`, `./data/simulation/splits/train_params_data.csv`
- test data: `./data/simulation/splits/test_motion_data.csv`, `./data/simulation/splits/test_params_data.csv`
- validation data: `./data/simulation/splits/val_motion_data.csv`, `./data/simulation/splits/val_params_data.csv`
にそれぞれ保存されます。

重力加速度や生成する軌道の数、train, test, valの比率などは以下のファイルを編集することで変更できます。
```
cfg/cfg.yaml
```

# ユニットテスト
コンテナ内で実行できるユニットテストとその内容を説明します。

放物線の生成をテストする。
```
make test_parabolic_motion_generation
```

放物線データの分割をテストする。
```
make test_parabolic_motion_split
```

## その他
特になし

_____

# Free-fall-surrogate

## Overview
This project aims to replicate free fall motion using a surrogate model.
- A surrogate model is a method that uses machine learning to replicate complex physical phenomena, which would typically require numerical analysis or simulations, in a simpler and faster manner.
- In this project, the goal is to predict free fall motion using a surrogate model, thereby improving the efficiency of physics-based simulations.
- The model calculates the trajectory of free fall based on two parameters: initial velocity and launch angle. The accuracy and applicability of the model are then evaluated.

## Development Environment
- OS: Ubuntu-20.04 (Docker container)
- Python: 3.10.10

## Installation
After installing Docker, you can build the image by running the following command, which references `./Dockerfile`:

```
make docker-build
```

You can start the container by running:
```
make docker-run
```
The container will have all the necessary dependencies installed and ready to use.

## Usage
Below are the commands that can be executed within the container, along with their descriptions.

Generate parabolic motion data:
```
make generate_parabolic_data
```
The generated data is saved in the following locations:
- `./data/simulation/parabolic_motion.csv` for the trajectory (time, x-coordinate, y-coordinate)
- `data/simulation/parabolic_params.csv` for the parameters such as initial velocity and launch angle

Split the generated parabolic motion data into train, test, and validation sets:
```
make split
```
The split data is saved in the following locations:
- Training data: `./data/simulation/splits/train_motion_data.csv`, `./data/simulation/splits/train_params_data.csv`
- Test data: `./data/simulation/splits/test_motion_data.csv`, `./data/simulation/splits/test_params_data.csv`
- Validation data: `./data/simulation/splits/val_motion_data.csv`, `./data/simulation/splits/val_params_data.csv`

You can modify parameters such as gravitational acceleration, the number of generated trajectories, and the ratios for train, test, and validation sets by editing the following file:
```
./cfg/cfg.yaml
```

# Unit Tests
Below are the unit tests that can be executed within the container, along with their descriptions.

Test the generation of parabolic motion:
```
make test_parabolic_motion_generation
```

Test the splitting of parabolic motion data:
```
make test_parabolic_motion_split
```

## Others
Nothing in particular.

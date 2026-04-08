# LucidTree

An implementation of Go AI, similar to [KataGo](https://github.com/lightvector/KataGo).

## Data/Model Download

For the processed and raw datasets, please check out this url on [Google Drive](https://drive.google.com/drive/folders/1Brh3DSuQ2fcs2gPFlytn4BvrR4j-qWHK?usp=sharing)

For the .pt model, please checkout this url on [Google Drive](https://drive.google.com/drive/folders/12OCXJz11Ely8U9kf6R822n4Apfg3QOef?usp=sharing)

## Sample Training Log

<details>
<summary>Click to Expand</summary>

```log
2026-03-02 14:19:52 | INFO | training | Starting training
2026-03-02 14:19:52 | INFO | training | Using device: cuda
2026-03-02 14:19:52 | INFO | training | CUDA enabled: Tesla T4 (device cuda)
2026-03-02 14:19:52 | INFO | training | Total epoch = 10
2026-03-02 14:19:52 | INFO | training | Board size = 19
2026-03-02 14:19:52 | INFO | training | Batch size = 256
2026-03-02 14:19:52 | INFO | training | Gradient accumulation steps = 1
2026-03-02 14:19:53 | WARNING | training | Checkpoint file does not exist. Starting with no checkpoint file.
2026-03-02 14:21:55 | INFO | training | train_dataset length: 11910180
2026-03-02 14:21:55 | INFO | training | val_dataset length: 1484562
2026-03-02 14:21:55 | INFO | training | test_dataset length: 1490073
2026-03-02 14:21:55 | INFO | training | Finished loading train_loader.
2026-03-02 14:21:55 | INFO | training | Finished loading val_loader.
2026-03-02 14:21:55 | INFO | training | Finished loading test_loader.
2026-03-02 14:21:55 | INFO | training | Start training loop.
2026-03-02 14:21:55 | INFO | training | Epoch 0 started. Total batches: 46525 (grad accum steps: 1).
2026-03-02 14:24:41 | INFO | training | Epoch 0 | Batch 0 | loss = 6.4552 | total_loss = 6.4552
2026-03-03 00:54:26 | INFO | training | Epoch 0 | Batch 1000 | loss = 3.5255 | total_loss = 3998.0189
2026-03-03 11:26:00 | INFO | training | Epoch 0 | Batch 2000 | loss = 3.5116 | total_loss = 7600.1766
2026-03-03 21:53:35 | INFO | training | Epoch 0 | Batch 3000 | loss = 3.3946 | total_loss = 11102.3919
2026-03-04 08:22:14 | INFO | training | Epoch 0 | Batch 4000 | loss = 3.2518 | total_loss = 14536.7324
2026-03-04 18:52:55 | INFO | training | Epoch 0 | Batch 5000 | loss = 3.4532 | total_loss = 17913.6715
2026-03-05 05:24:18 | INFO | training | Epoch 0 | Batch 6000 | loss = 3.4546 | total_loss = 21251.8446
2026-03-05 15:57:44 | INFO | training | Epoch 0 | Batch 7000 | loss = 2.9904 | total_loss = 24553.4064
2026-03-06 02:26:40 | INFO | training | Epoch 0 | Batch 8000 | loss = 3.3198 | total_loss = 27835.5125
2026-03-06 12:56:47 | INFO | training | Epoch 0 | Batch 9000 | loss = 3.2537 | total_loss = 31088.1558
2026-03-06 23:28:27 | INFO | training | Epoch 0 | Batch 10000 | loss = 3.0867 | total_loss = 34325.1487
2026-03-07 09:57:54 | INFO | training | Epoch 0 | Batch 11000 | loss = 3.1091 | total_loss = 37540.0646
2026-03-07 20:27:09 | INFO | training | Epoch 0 | Batch 12000 | loss = 3.3522 | total_loss = 40732.7597
2026-03-08 06:56:17 | INFO | training | Epoch 0 | Batch 13000 | loss = 3.1251 | total_loss = 43916.3182
2026-03-08 17:26:02 | INFO | training | Epoch 0 | Batch 14000 | loss = 2.8750 | total_loss = 47082.0043
2026-03-09 03:55:07 | INFO | training | Epoch 0 | Batch 15000 | loss = 2.9189 | total_loss = 50228.6539
2026-03-09 14:24:48 | INFO | training | Epoch 0 | Batch 16000 | loss = 3.0630 | total_loss = 53356.6642
2026-03-10 00:55:10 | INFO | training | Epoch 0 | Batch 17000 | loss = 3.1561 | total_loss = 56473.7347
2026-03-10 11:26:21 | INFO | training | Epoch 0 | Batch 18000 | loss = 3.0193 | total_loss = 59570.3821
2026-03-10 21:55:32 | INFO | training | Epoch 0 | Batch 19000 | loss = 3.1720 | total_loss = 62658.3242
2026-03-11 08:25:20 | INFO | training | Epoch 0 | Batch 20000 | loss = 3.0200 | total_loss = 65737.8792
2026-03-11 18:53:50 | INFO | training | Epoch 0 | Batch 21000 | loss = 2.9343 | total_loss = 68793.9651
2026-03-12 05:23:29 | INFO | training | Epoch 0 | Batch 22000 | loss = 3.0230 | total_loss = 71840.4682
2026-03-12 15:54:07 | INFO | training | Epoch 0 | Batch 23000 | loss = 3.1741 | total_loss = 74879.6330
2026-03-13 02:22:23 | INFO | training | Epoch 0 | Batch 24000 | loss = 3.0876 | total_loss = 77904.2730
2026-03-13 12:57:30 | INFO | training | Epoch 0 | Batch 25000 | loss = 2.9856 | total_loss = 80919.5121
2026-03-13 23:28:49 | INFO | training | Epoch 0 | Batch 26000 | loss = 3.1521 | total_loss = 83922.1933
2026-03-14 09:58:00 | INFO | training | Epoch 0 | Batch 27000 | loss = 3.1709 | total_loss = 86916.3163
2026-03-14 20:26:12 | INFO | training | Epoch 0 | Batch 28000 | loss = 2.9535 | total_loss = 89904.3535
2026-03-15 06:55:18 | INFO | training | Epoch 0 | Batch 29000 | loss = 2.8643 | total_loss = 92879.9853
2026-03-15 17:24:51 | INFO | training | Epoch 0 | Batch 30000 | loss = 2.8073 | total_loss = 95836.0552
2026-03-16 03:53:55 | INFO | training | Epoch 0 | Batch 31000 | loss = 2.9543 | total_loss = 98789.0316
2026-03-16 14:22:48 | INFO | training | Epoch 0 | Batch 32000 | loss = 2.9952 | total_loss = 101733.0259
2026-03-17 00:52:50 | INFO | training | Epoch 0 | Batch 33000 | loss = 2.8964 | total_loss = 104664.6101
2026-03-17 11:21:26 | INFO | training | Epoch 0 | Batch 34000 | loss = 3.0877 | total_loss = 107591.2003
2026-03-17 21:49:56 | INFO | training | Epoch 0 | Batch 35000 | loss = 2.7083 | total_loss = 110512.6806
2026-03-18 08:17:47 | INFO | training | Epoch 0 | Batch 36000 | loss = 2.8176 | total_loss = 113419.9389
2026-03-18 18:45:55 | INFO | training | Epoch 0 | Batch 37000 | loss = 2.9288 | total_loss = 116326.2291
2026-03-19 05:15:05 | INFO | training | Epoch 0 | Batch 38000 | loss = 3.0942 | total_loss = 119220.8333
2026-03-19 15:44:53 | INFO | training | Epoch 0 | Batch 39000 | loss = 2.8944 | total_loss = 122110.3280
2026-03-20 02:15:51 | INFO | training | Epoch 0 | Batch 40000 | loss = 2.9244 | total_loss = 124985.5085
2026-03-20 12:47:23 | INFO | training | Epoch 0 | Batch 41000 | loss = 2.9609 | total_loss = 127858.7117
2026-03-20 23:18:03 | INFO | training | Epoch 0 | Batch 42000 | loss = 2.8284 | total_loss = 130725.0095
2026-03-21 09:45:58 | INFO | training | Epoch 0 | Batch 43000 | loss = 2.8915 | total_loss = 133585.1391
2026-03-21 20:14:24 | INFO | training | Epoch 0 | Batch 44000 | loss = 2.7208 | total_loss = 136435.7427
2026-03-22 06:42:28 | INFO | training | Epoch 0 | Batch 45000 | loss = 2.9194 | total_loss = 139281.2409
2026-03-22 17:11:05 | INFO | training | Epoch 0 | Batch 46000 | loss = 2.8271 | total_loss = 142117.2707
2026-03-25 12:17:49 | INFO | training | Found a better state at epoch 0
2026-03-25 12:17:49 | INFO | training | Epoch 0 finished | train_loss = 3.0866 | val_loss = 3.1268 | val_acc1 = 0.4325 | val_acc5 = 0.7619
```

</details>

## Timeline

### Week 1

Implemented basic Go board engine and a simple minimax file for tac-tac-toe that will be later used for Go as a depth-limited MiniMax (and probably see it fails badly)

New features include:

- Place move at specific position with specific color
- Captures detection
- Ko detection
- Score estimation at the end of the game
- Illegal move detection
- Display a real Go board with MatPlotLib

### Week 2

Implemented a basic depth-limited MiniMax algorithm for Go with alpha-beta pruning. It checks all possible moves in a given board state and choose the local optimal one by choosing the move that captures the most opponent's stones. Also did some minor updates to the board class.

New features include:

- Depth-limited Minimax algorithm with alpha-beta pruning
- Auto game-over when there are 2 consecutive passes
- Undo feature for game board

### Week 3 + 4

Implemented a basic Monte Carlo Tree Search for Go, as well as a Node class. The algorithm works by randomly choose legal position to play and calculate the UCT (Upper Confidence Bound applied to Trees). At the end, it picks the node with the most visits to ensure stability.

New features include:

- A basic Monte Carlo Tree Search for Go
- A new Node class data structure

### Week 5 + 6 + 7

Implemented a Convolution Neural Network (CNN) for Go with PyTorch. It works along with a policy network and a value network that allows the MCTS to perform better searching. Also refactored file structure so it's more sorted.

New features include:

- A decent Neural Network that learn from over 400 9\*9 Go .sgf files.
- Comprehensive logging when training
- Pre-computed dataset
- Model auto-saving

### Week 8

Combined Monte Carlo Tree Search with Neural Network, similar to how AlphaZero works. It uses a PUCT (prior upper confidence score for trees) score instead of the ordinary UCT, (or "UCB"), in order to balance exploration and exploitation.

New features include:

- Combination of Monte Carlo Tree Search and Neural Network
- Stronger NN with more datasets
- Minor bug fixes for board.py

### Week 9 + 10 + 11

Improved Monte Carlo Tree Search to ensure that there is no logical errors. Moved training process to AWS EC2 for better efficiency and memory.

New features include:

- Optimized Monte Carlo Tree Search
- Better training settings for CUDA GPU

## API Architecture

Implemented a functional Django Rest Framework API in `/api` directory. It receives JSON input and send back a JSON output with the best move.

<details>
<summary>Sample Request</summary>

```json
{
    "board_size": 9,
    "rules": "japanese",
    "komi": 6.5,
    "to_play": "B",
    "moves": [
        ["B", "D4"],
        ["W", "D16"]
    ],
    "algo": "mcts",
    "params": {
        "num_simulations": 500,
        "c_puct": 1.25,
        "max_time_ms": 1000,
        "temperature": 0.0,
        "random_seed": 0,
        "select_by": "visit_count"
    },
    "output": {
        "include_top_moves": 5,
        "include_policy": true,
        "include_winrate": true,
        "include_visits": true
    }
}
```

</details>

<details>
<summary>Sample Response</summary>

```json
{
    "top_moves": [
        {
            "move": "Q3",
            "policy": 0.3518323600292206,
            "winrate": 0.03272856026887894,
            "visits": 56
        },
        {
            "move": "Q17",
            "policy": 0.15180940926074982,
            "winrate": 0.027093904092907906,
            "visits": 25
        },
        {
            "move": "R4",
            "policy": 0.13219143450260162,
            "winrate": 0.01938430592417717,
            "visits": 21
        },
        {
            "move": "R16",
            "policy": 0.12820260226726532,
            "winrate": 0.01963173970580101,
            "visits": 21
        },
        {
            "move": "Q4",
            "policy": 0.10890644043684006,
            "winrate": 0.030727697536349297,
            "visits": 16
        }
    ],
    "algorithm": "mcts",
    "stats": {
        "model": "checkpoint_19x19",
        "num_simulations": 500,
        "c_puct": 1.25,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon": 0.0,
        "value_weight": 1.0,
        "policy_weight": 1.0,
        "select_by": "visit_count",
        "include_visits": true,
        "simulations_run": 152,
        "max_time_ms": 1000,
        "policy": [
            8.48091731313616e-5, 0.00011868352157762274, 9.970022074412555e-5,
            0.00010437508171889931, 0.00010991421004291624,
            9.189730189973488e-5, 8.829159924061969e-5, 8.148477354552597e-5,
            7.697017281316221e-5, 7.801960600772873e-5, 7.818383892299607e-5,
            8.145595347741619e-5, 8.59393403516151e-5, 0.00010544621909502894,
            0.00013809437223244458, 0.00011695632565533742,
            8.254143176600337e-5, 7.524532702518627e-5, 9.242323721991852e-5,
            8.257347508333623e-5, 8.103453001240268e-5, 7.685984746785834e-5,
            0.0001048825797624886, 0.00010394576383987442, 8.335327584063634e-5,
            8.082351996563375e-5, 7.773278048262e-5, 7.308540807571262e-5,
            7.472764991689473e-5, 7.285228639375418e-5, 7.590306631755084e-5,
            8.032344339881092e-5, 0.00010009100515162572, 9.168387623503804e-5,
            7.495462341466919e-5, 8.157597767421976e-5, 8.23539012344554e-5,
            8.843359682941809e-5, 8.371367584913969e-5, 8.0644509580452e-5,
            7.368201477220282e-5, 8.635925041744485e-5, 8.756091119721532e-5,
            0.0001352237450191751, 8.96780620678328e-5, 8.422601968050003e-5,
            9.187066461890936e-5, 0.00010675920930225402,
            0.00010251337516820058, 0.0001272595691261813,
            0.00019579681975301355, 0.0011820943327620625, 0.010151753202080727,
            0.3518323600292206, 0.00027563105686567724, 5.4281546908896416e-5,
            9.38984812819399e-5, 8.909897587727755e-5, 9.101673640543595e-5,
            0.00010098547500092536, 0.00014368303527589887,
            0.00010136564378626645, 0.00010102188389282674,
            7.297041884157807e-5, 6.188886618474498e-5, 6.803101859986782e-5,
            0.00014516316878143698, 0.00015660384087823331,
            0.00013248540926724672, 0.00013313903764355928,
            0.0007714091916568577, 0.006393783260136843, 0.10890644043684006,
            0.13219143450260162, 4.393800190882757e-5, 7.894160808064044e-5,
            8.817371417535469e-5, 8.938826067605987e-5, 8.550564962206408e-5,
            0.000125785285490565, 0.00010980598744936287, 8.420521771768108e-5,
            8.301904745167121e-5, 8.60142899909988e-5, 8.421742677455768e-5,
            8.378156053368002e-5, 8.608624921180308e-5, 9.30840105866082e-5,
            0.00010639523679856211, 0.00013342121383175254,
            0.00016664114082232118, 0.0025803425814956427,
            0.0012387220049276948, 5.424102710094303e-5, 6.740504613844678e-5,
            8.102502761175856e-5, 8.400694059673697e-5, 9.294491610489786e-5,
            8.750014239922166e-5, 7.711660146014765e-5, 7.234892837004736e-5,
            7.276763790287077e-5, 7.984648254932836e-5, 7.47939629945904e-5,
            7.04536578268744e-5, 7.201787957455963e-5, 7.352617831202224e-5,
            7.696921966271475e-5, 8.629306103102863e-5, 9.713244071463123e-5,
            0.00012038346176268533, 0.00010989953443640843, 7.24987403373234e-5,
            7.673227082705125e-5, 7.882372301537544e-5, 9.409309132024646e-5,
            9.053307439899072e-5, 8.931711636250839e-5, 8.048001473071054e-5,
            8.529674960300326e-5, 9.003935701912269e-5, 7.909646956250072e-5,
            7.163237751228735e-5, 7.117782661225647e-5, 7.638575334567577e-5,
            7.65992735978216e-5, 7.709013152634725e-5, 7.748121424810961e-5,
            9.146991214947775e-5, 0.00010171405301662162, 7.822903717169538e-5,
            8.368230191990733e-5, 8.538481051800773e-5, 8.192165842046961e-5,
            8.453092596028e-5, 7.310847286134958e-5, 7.144891424104571e-5,
            8.146115578711033e-5, 8.093976066447794e-5, 8.675854041939601e-5,
            7.962208474054933e-5, 7.412123522954062e-5, 7.620079850312322e-5,
            8.163606253219768e-5, 8.089901530183852e-5, 8.157100091921166e-5,
            7.693222869420424e-5, 8.848328434396535e-5, 9.034160029841587e-5,
            7.30281972209923e-5, 8.426072599831969e-5, 8.561017602914944e-5,
            8.795006579020992e-5, 7.184172136476263e-5, 7.356798596447334e-5,
            7.516342884628102e-5, 7.864832150517032e-5, 7.986453420016915e-5,
            8.511594933224842e-5, 8.444864943157881e-5, 7.844756328267977e-5,
            7.969790021888912e-5, 8.022353722481057e-5, 8.058945968514308e-5,
            8.220160088967532e-5, 7.955938781378791e-5, 9.1453519416973e-5,
            9.192692959913984e-5, 7.221622945507988e-5, 8.563312439946458e-5,
            8.55352554935962e-5, 8.458745287498459e-5, 8.423373219557106e-5,
            9.803841385291889e-5, 7.728608761681244e-5, 7.793817349011078e-5,
            8.23775480967015e-5, 8.432270988123491e-5, 8.302609785459936e-5,
            8.327167597599328e-5, 8.024480484891683e-5, 7.769364310661331e-5,
            7.870324043324217e-5, 8.142861042870209e-5, 7.993745384737849e-5,
            9.274375042878091e-5, 9.285472333431244e-5, 7.186186849139631e-5,
            8.554210944566876e-5, 8.493795758113265e-5, 8.228089427575469e-5,
            8.210547093767673e-5, 0.00012377926032058895, 9.165424853563309e-5,
            8.807404083199799e-5, 8.023731061257422e-5, 8.384574903175235e-5,
            8.069151226663962e-5, 8.812789747025818e-5, 8.281045302283019e-5,
            7.742957677692175e-5, 7.7921969932504e-5, 8.094694203464314e-5,
            7.962588279042393e-5, 9.328511805506423e-5, 9.295901691075414e-5,
            7.120417285477743e-5, 8.460455865133554e-5, 8.522990538040176e-5,
            8.637745486339554e-5, 7.782037573633716e-5, 0.00010014534200308844,
            0.00010976378689520061, 9.854967356659472e-5, 8.586020703660324e-5,
            8.617562707513571e-5, 8.690362301422283e-5, 8.869813609635457e-5,
            8.361760410480201e-5, 7.869835826568305e-5, 7.62255149311386e-5,
            7.990300218807533e-5, 7.978368375916034e-5, 9.264624532079324e-5,
            0.0001014002482406795, 7.758480933262035e-5, 8.398147474508733e-5,
            8.409952715737745e-5, 8.476688526570797e-5, 8.547059405827895e-5,
            9.859845886239782e-5, 0.0001239375997101888, 0.00010264779120916501,
            9.413132647750899e-5, 8.669254020787776e-5, 9.249051799997687e-5,
            8.906533184926957e-5, 8.571302896598354e-5, 8.159270510077477e-5,
            7.95433807070367e-5, 8.01871283329092e-5, 8.133734081638977e-5,
            9.48023225646466e-5, 9.941386815626174e-5, 9.301736281486228e-5,
            8.027673175092787e-5, 8.098623948171735e-5, 7.45956422179006e-5,
            0.0001051291183102876, 0.002829889999702573, 0.00016253464855253696,
            8.590165089117363e-5, 8.771553984843194e-5, 9.115563443629071e-5,
            8.633965626358986e-5, 8.831476588966325e-5, 8.868274744600058e-5,
            8.729052933631465e-5, 8.499118848703802e-5, 8.586553303757682e-5,
            8.720772893866524e-5, 0.00011116266250610352, 0.0001854958973126486,
            0.00014672806719318032, 8.256481669377536e-5, 7.74504806031473e-5,
            8.936200174503028e-5, 0.00012416178651619703,
            0.00012023218005197123, 0.00013039709301665425,
            8.818187052384019e-5, 0.000109705506474711, 8.840711234370247e-5,
            8.219925075536594e-5, 8.766685641603544e-5, 8.982665167422965e-5,
            8.674572018207982e-5, 8.595434337621555e-5, 9.122285700868815e-5,
            9.714365296531469e-5, 0.00017150412895716727, 0.0008859917870722711,
            0.0021561605390161276, 6.792152998968959e-5, 6.902707536937669e-5,
            7.930940046207979e-5, 0.00011909021122846752,
            0.00022173899924382567, 8.235649147536606e-5, 9.737970685819164e-5,
            0.00010193604248343036, 9.693495667306706e-5, 8.09566699899733e-5,
            8.830532897263765e-5, 8.989306661533192e-5, 8.735314622754231e-5,
            8.703384082764387e-5, 9.349441825179383e-5, 0.00012753768533002585,
            0.0007127823191694915, 0.0565904900431633, 0.12820260226726532,
            4.2095645767403767e-5, 7.226403977256268e-5, 8.276939479401335e-5,
            0.0001358280424028635, 0.00010790584201458842,
            0.00011963702127104625, 0.00010106332774739712,
            0.0017697399016469717, 0.00010204059799434617, 7.377327710855752e-5,
            7.873641879996285e-5, 8.927810267778113e-5, 9.873462113318965e-5,
            0.000112218338472303, 0.00012491508095990866, 0.0004638792888727039,
            0.008200022391974926, 0.15180940926074982, 0.0002534640079829842,
            5.173844692762941e-5, 7.218358950922266e-5, 5.536635217140429e-5,
            8.296910527860746e-5, 7.888276013545692e-5, 0.0001031172796501778,
            0.00012607622193172574, 8.521178096998483e-5, 8.193095709430054e-5,
            7.557882054243237e-5, 7.255843229359016e-5, 7.544134859926999e-5,
            7.619004463776946e-5, 7.6735632319469e-5, 7.795891724526882e-5,
            0.00010189999011345208, 8.518261165590957e-5, 6.780284456908703e-5,
            6.723734986735508e-5, 5.3065396059537306e-5, 7.539740181528032e-5,
            0.00017588966875337064, 0.0001033274456858635, 9.854008385445923e-5,
            0.00010804176417877898, 0.00011014579649781808,
            0.00010742335871327668, 9.535215940559283e-5, 7.49261089367792e-5,
            6.898331048432738e-5, 7.051724242046475e-5, 7.283018931047991e-5,
            7.738461863482371e-5, 8.469472959404811e-5, 0.00010082794324262068,
            0.00010086497059091926, 8.463990525342524e-5, 7.114755862858146e-5,
            7.367787475232035e-5, 7.574159826617688e-5, 7.79554175096564e-5
        ],
        "winrate": 0.016697337850928307,
        "elapsed_ms": 1066.44
    }
}
```

</details>

## Src File structure

```yaml
LucidTree/
├── src/
│   ├── lucidtree/
│   │   │── cli
│   │   │   ├── main.py                     # Python file for testing
│   │   │── common
│   │   │   ├── logging.py                  # Logger setup
│   │   │   ├── paths.py                    # Paths-related functions, such as getting project root
│   │   │── engine
│   │   │   ├── analysis.py                 # Function that handles a validated JSON input analysis request
│   │   │── go
│   │   │   ├── board.py                    # Python class that represents a Go game board
│   │   │   ├── game.py                     # Python class that represents a Go game, including board, players, and the winner
│   │   │   ├── move.py                     # Python class that represents a move in a game of Go
│   │   │   ├── player.py                   # Python class that represents a player in a game of Go
│   │   │   ├── rules.py                    # Python class that contains various rules for Go
│   │   │── mcts
│   │   │   ├── node.py                     # A custom Node data structure class used for Monte Carlo Tree Search
│   │   │   ├── search.py                   # A python program that searches for the most optimum move given the board and player
│   │   │── minimax
│   │   │   ├── search.py                   # A depth-limited MiniMax algorithm for Go with alpha-beta pruning
│   │   │── nn                              # All neural network related files
│   │   │   ├── datasets/
│   │   │   │   ├── gokifu_download.py      # A Python program that automatically downloads professional games from Gokifu website
│   │   │   │   ├── precomputed_dataset.py  # A class that represents a pre-computed dataset
│   │   │   │   ├── sgf_dataset.py          # A one-time running file that generates all the datasets
│   │   │   │   ├── sgf_parser.py           # An util file that parses SGF files and convert it to a Game object
│   │   │   ├── agent.py                    # The agent that loads the model and pick a move
│   │   │   ├── evaluate.py                 # A function that evaluate the training result based on the validation dataset
│   │   │   ├── features.py                 # Some features that are related to nn
│   │   │   ├── model.py                    # The PolicyValueNetwork CNN model
│   │   │   ├── split.py                    # Splits the game into training, validation, and testing set
│   │   │   ├── train.py                    # Runs the actual training with 30 epochs
│   │   │── constants.py                    # A file containing all the essential constants used in the project
```

## Django Rest Framework API

LucidTree uses Django Rest Framework (DRF) as its backend framework. All related files are located in the `/api` directory. To start the server, first follow the [development setup](#development) tutorial, then type the following command to start a local development server:

```bash
python manage.py makemigrations && python manage.py migrate
python manage.py runserver
```

## Development

To start developing this project locally. Run the following command:

Install UV:

```bash
uv --version  # check if UV is already installed

# If it is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # MacOS & Linux
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

Clone the repository and setup:

```bash
git clone https://github.com/YianXie/LucidTree  # Clone this repository
cd LucidTree
uv init  # initialize the virtual environment
uv sync --dev  # install the dependencies
source .venv/bin/activate  # activate the virtual environment
```

Now you are ready to start developing. To see a quick demo, you may go the `main.py` and try a few different .sgf files or play your own.

> [!NOTE]
> Note: in some cases, the `main.py` file may not run correctly due to the `Module Not Found` error. In that case, try running the `make run` command at root level.

If there are any issues while developing, feel free to create an issue under the `issues` tab in the GitHub repository page.

Happy developing!

## Tests

This project contains some tests that you can run while developing to make sure everything works as expected.

To run tests:

```bash
uv init  # initialize the virtual environment if you haven't already done it
uv sync --dev  # install all the dependencies
```

```bash
make test  # Run at root level. This would run all tests.
# or
pytest  # Directly call the pytest command
```

To add more tests, simply add a new Python file in the `tests/` directory. Note that it must start with `test_xxx` or `xxx_test`

## CI/CD

This project uses GitHub Actions for continuous integration. The `ci.yml` workflow runs on every push and pull request, performing code quality checks including Ruff linting, Mypy type checking, isort import sorting validation, and pip-audit security scanning. The `tests.yml` workflow runs pytest tests on pushes to the main branch and all pull requests targeting main, ensuring that all tests pass before code is merged.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

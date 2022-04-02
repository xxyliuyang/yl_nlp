local model_name = "../resources/bert-base-cased";
local train_data = './data/data4bert/nyt/demo.json';
local dev_data = './data/data4bert/nyt/demo.json';
local rel2id_file = './data/data4bert/nyt/rel2id.json';

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,

    "dataset_reader": {
        "type": "tplinker",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "add_special_tokens":false,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "tokens",
            }
        },
        "rel2id_file": rel2id_file,
        "max_tokens": 300,
    },

    "pytorch_seed": 42,
    "numpy_seed": 42,
    "random_seed": 42,

    "model": {
        "type": "tplinker",
        "model_name": model_name,
        "rel2id_file": rel2id_file,
    },

    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 32,
        },
    },

    "validation_data_loader": {
        "batch_size": 32,
        "shuffle": false
    },

    "trainer": {
        "num_epochs": 50,
        "use_amp": false,
        "num_gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay":0.01,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac":0.1
        },
        "grad_norm":1.0,
        "cuda_device":1,
    }
}

import torch
import wandb
import numpy as np
from tqdm import tqdm
from utils.basic_utils import Logger
from metric.predict_metric import PredictMetric


def _calc_Recall(sort_lists, batch_size, topk=10):
    Recall_result = torch.sum(sort_lists < topk) / batch_size
    return Recall_result


def _calc_NDCG(sort_lists, batch_size, topk=10):
    hit = sort_lists < topk
    NDCG_score = hit * (1 / torch.log2(sort_lists + 2))
    NDCG_result = torch.sum(NDCG_score) / batch_size
    return NDCG_result


class Evaluator(object):
    def __init__(self,
                 args,
                 fusion_model,
                 eval_dataset,
                 eval_dataloader,
                 test_dataset,
                 test_dataloader,
                 **kwargs):
        self.args = args
        self.device = args.device
        self.logger = Logger(__name__).get_logger(args.log_file)

        self.fusion_model = fusion_model

        self.eval_dataset = eval_dataset
        self.eval_dataloader = eval_dataloader
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader
        self.metric = PredictMetric()

    @torch.no_grad()
    def eval(self, epoch):
        self.fusion_model.eval()

        self.logger.info(f"***** Run eval *****")
        sort_lists = []
        batch_size = 0

        for step, data in tqdm(enumerate(self.eval_dataloader)):
            input_ids = data.to(self.device)
            print("input", input_ids)
            pred, label = self.fusion_model(input_ids, mode="pred")
            print("pred", pred)
            print("label", label)
            sort_index, batch = self.metric(pred, label)
            print("sort idx", sort_index)
            print("batch", batch)
            sort_lists.append(sort_index)
            batch_size += batch

        sort_lists = torch.cat(sort_lists, dim=0)

        Recall10 = _calc_Recall(sort_lists, batch_size, 10)
        Recall50 = _calc_Recall(sort_lists, batch_size, 50)
        NDCG10 = _calc_NDCG(sort_lists, batch_size, 10)
        NDCG50 = _calc_NDCG(sort_lists, batch_size, 50)

        if self.args.wandb_enable:
            wandb.log({"eval/Recall@10": Recall10,
                       "eval/Recall@50": Recall50,
                       "eval/NDCG@10": NDCG10,
                       "eval/NDCG@50": NDCG50,
                       "train/epoch": epoch})
        self.logger.info(f"Epoch {epoch} Eval Result: R@10:{Recall10}, R@50:{Recall50}, NDCG@10:{NDCG10}, NDCG@50:{NDCG50}")

        return Recall10

    @torch.no_grad()
    def test(self):
        self.fusion_model.eval()

        self.logger.info(f"***** Run test *****")
        sort_lists = []
        batch_size = 0

        for step, data in tqdm(enumerate(self.test_dataloader)):
            input_ids = data.to(self.device)
            pred, label = self.fusion_model(input_ids, mode="pred")
            sort_index, batch = self.metric(pred, label)

            sort_lists.append(sort_index)
            batch_size += batch

        sort_lists = torch.cat(sort_lists, dim=0)

        Recall10 = _calc_Recall(sort_lists, batch_size, 10)
        Recall50 = _calc_Recall(sort_lists, batch_size, 50)
        NDCG10 = _calc_NDCG(sort_lists, batch_size, 10)
        NDCG50 = _calc_NDCG(sort_lists, batch_size, 50)

        if self.args.wandb_enable:
            wandb.log({"test/Recall@10": Recall10,
                       "test/Recall@50": Recall50,
                       "test/NDCG@10": NDCG10,
                       "test/NDCG@50": NDCG50})
        self.logger.info(f"Test Result: R@10:{Recall10}, R@50:{Recall50}, NDCG@10:{NDCG10}, NDCG@50:{NDCG50}")

    def calc_res(self, scores, tst_ids, pos_ids, seq_len):
        group_h20 = [0] * 4
        group_n20 = [0] * 4
        group_num = [0] * 4
        h5, n5, h10, n10, h20, n20, h50, n50 = [0] * 8
        for i in range(len(pos_ids)):
            ids_with_scores = list(zip(tst_ids[i], scores[i]))
            ids_with_scores = sorted(ids_with_scores, key=lambda x: x[1], reverse=True)
            if seq_len[i] < 5:
                group_num[0] += 1
            elif seq_len[i] >= 5 and seq_len[i] < 10:
                group_num[1] += 1
            elif seq_len[i] >= 10 and seq_len[i] < 20:
                group_num[2] += 1
            else:
                group_num[3] += 1
            shoot = list(map(lambda x: x[0], ids_with_scores[:5]))
            if pos_ids[i] in shoot:
                h5 += 1
                n5 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            shoot = list(map(lambda x: x[0], ids_with_scores[:10]))
            if pos_ids[i] in shoot:
                h10 += 1
                n10 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            shoot = list(map(lambda x: x[0], ids_with_scores[:20]))
            if pos_ids[i] in shoot:
                if seq_len[i] < 5:
                    group_h20[0] += 1
                    group_n20[0] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                elif seq_len[i] >= 5 and seq_len[i] < 10:
                    group_h20[1] += 1
                    group_n20[1] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                elif seq_len[i] >= 10 and seq_len[i] < 20:
                    group_h20[2] += 1
                    group_n20[2] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                else:
                    group_h20[3] += 1
                    group_n20[3] += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
                h20 += 1
                n20 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
            shoot = list(map(lambda x: x[0], ids_with_scores[:50]))
            if pos_ids[i] in shoot:
                h50 += 1
                n50 += np.reciprocal(np.log2(shoot.index(pos_ids[i]) + 2))
        return h5, n5, h10, n10, h20, n20, h50, n50, group_h20, group_n20, group_num


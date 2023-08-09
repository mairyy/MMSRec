import torch
import wandb
from tqdm import tqdm
from utils.basic_utils import Logger
from metric.predict_metric import PredictMetric

def _calc_HitRatio(score, seq_lists, topk=10):
    num_seq = seq_lists.shape[0]
    hr = 0
    for i in range(num_seq):
        if seq_lists[i] in score[:topk]:
            hr += 1
    return hr

def _calc_Recall(sort_lists, batch_size, topk=10):
    Recall_result = torch.sum(sort_lists < topk) / batch_size
    return Recall_result

def _calc_NDCG(sort_lists, batch_size, topk=10):
    hit = sort_lists < topk
    NDCG_score = hit * (1 / torch.log2(sort_lists + 2))
    NDCG_result = torch.sum(NDCG_score) / batch_size
    return NDCG_result

def _calc_HR(pred, seq, topk=10):
    print()
    pred = pred[:topk]
    hr = 0
    HR_result = 0
    for i in range(pred.shape[0]):
        for idx in range(topk):
            if pred[i, idx] in seq[i]:
                hr += 1
        HR_result += hr / seq[i].shape[0]
        hr = 0
    HR_result /= seq.shape[0]
    return HR_result

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
        HR = [0] * 3

        for step, data in tqdm(enumerate(self.eval_dataloader)):
            input_ids = data.to(self.device)
            # print("len", input_ids.shape[0], "input", input_ids)
            pred, label = self.fusion_model(input_ids, mode="pred")
            # print("len", pred.shape[0], "pred", pred)
            # print("len", label.shape[0], "label", label)
            pred_sort, sort_index, batch = self.metric(pred, label, input_ids)
            # print("len", sort_index.shape[0], "sort idx", sort_index)
            # print("batch", batch)
            sort_lists.append(sort_index)
            batch_size += batch
            HR[0] += _calc_HitRatio(pred_sort, input_ids, 5)
            HR[1] += _calc_HitRatio(pred_sort, input_ids, 10)
            HR[2] += _calc_HitRatio(pred_sort, input_ids, 20)

        sort_lists = torch.cat(sort_lists, dim=0)

        HR5 = HR[0] / batch_size
        HR10 = HR[1] / batch_size
        HR20 = HR[2] / batch_size
        Recall10 = _calc_Recall(sort_lists, batch_size, 10)
        Recall50 = _calc_Recall(sort_lists, batch_size, 50)
        NDCG5 = _calc_NDCG(sort_lists, batch_size, 5)
        NDCG10 = _calc_NDCG(sort_lists, batch_size, 10)
        NDCG20 = _calc_NDCG(sort_lists, batch_size, 20)
        NDCG50 = _calc_NDCG(sort_lists, batch_size, 50)

        if self.args.wandb_enable:
            wandb.log({"eval/HR@5": HR5,
                       "eval/HR@10": HR10,
                       "eval/HR@20": HR20,
                       "eval/Recall@10": Recall10,
                       "eval/Recall@50": Recall50,
                       "eval/NDCG@5": NDCG5,
                       "eval/NDCG@10": NDCG10,
                       "eval/NDCG@20": NDCG20,
                       "eval/NDCG@50": NDCG50,
                       "train/epoch": epoch})
        self.logger.info(f"Epoch {epoch} Eval Result: HR@5:{HR5}, HR@10:{HR10}, HR@20:{HR20}, R@10:{Recall10}\
                         , R@50:{Recall50}, NDCG@5:{NDCG5}, NDCG@10:{NDCG10}, NDCG@20:{NDCG20}, NDCG@50:{NDCG50}")

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


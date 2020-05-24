import os
import torch


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def save_result(TASK, model, EPOCH, yvalid, predvalid, losses, attentions):
    save_file_dir = os.path.join("result", TASK, repr(model))
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    save_file_path = os.path.join(save_file_dir,"epoch" + str(EPOCH) + ".pth")
    report = {
        "y": yvalid,
        "pred": predvalid,
        "loss": losses,
        "attentions": attentions
    }
    torch.save(report, save_file_path)

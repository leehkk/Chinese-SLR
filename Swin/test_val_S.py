import torch
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

def val_epoch(model, criterion, dataloader, device, epoch, logger, writer, weight_path=None, phase="Val"):
    if weight_path == None and phase == "Test":
        logger.info("failed to load weights,use k400 weights")
    elif weight_path != None:
        model.load_state_dict(torch.load(weight_path))
    model.eval()
    losses = []
    all_label = []
    all_pred = []
    val_5 = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            for i in range(outputs.shape[0]):
                top_k_values, top_k_indices = torch.topk(outputs[i], k=5)
                top_5 = []
                for j in range(5):
                    top_5.append(int(top_k_indices[j]))
                if int(labels[i].item()) in top_5:
                    val_5 += 1
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
            score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(),
                                   prediction.cpu().data.squeeze().numpy())
            if phase == "Test":
                logger.info(
                    "Test iteration {} Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(batch_idx, epoch + 1, loss,
                                                                                             score * 100))
            else:
                logger.info("Validation iteration {} Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(batch_idx, epoch + 1, loss,
                                                                                            score * 100))
        # Compute the average loss & accuracy
        validation_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        validation_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    # if phase == 'Test':
    #     with open('{}/swin_epoch050.pkl'.format(exp_name), 'wb') as f:
    #         score_dict = dict(zip(dataloader.dataset.labels, score))
    #         pickle.dump(score_dict, f)
    # Log
    if phase == 'Test':
        writer.add_scalars('Loss', {'Test': validation_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Test': validation_acc}, epoch + 1)
        logger.info("Average Test Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | Top5_Acc: {:.2f}".format(epoch + 1, validation_loss,
                                                                                        validation_acc * 100, val_5/len(all_label)))
    else:
        writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
        writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
        logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | Top5_Acc: {:.2f}".format(epoch+1, validation_loss, validation_acc*100, val_5/len(all_label)))
    return validation_loss
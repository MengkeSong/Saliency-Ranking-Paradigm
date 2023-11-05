rankloss = 0
for i in range(output.shape[0]):
    if i ==0:
        for j in [1,2,3,4]:
            propsal1 = output[i]
            propsal2 = output[j]
            propsal_lab1 = labels[i]
            propsal_lab2 = labels[j]
            if propsal_lab1 > propsal_lab2:
                target = torch.ones(1)
                target = target.cuda()
            else:
                target = torch.tensor([-1])
                target = target.cuda()

            loss_rank = nn.functional.margin_ranking_loss(propsal1, propsal2, target=target, margin=0)
            rankloss += loss_rank
    if i == 1:
        for j in [0, 2, 3, 4]:
            propsal1 = output[i]
            propsal2 = output[j]
            propsal_lab1 = labels[i]
            propsal_lab2 = labels[j]
            if propsal_lab1 > propsal_lab2:
                target = torch.ones(1)
                target = target.cuda()
            else:
                target = torch.tensor([-1])
                target = target.cuda()

            loss_rank = nn.functional.margin_ranking_loss(propsal1, propsal2, target=target, margin=0)
            rankloss += loss_rank
    if i == 2:
        for j in [0, 1, 3, 4]:
            propsal1 = output[i]
            propsal2 = output[j]
            propsal_lab1 = labels[i]
            propsal_lab2 = labels[j]
            if propsal_lab1 > propsal_lab2:
                target = torch.ones(1)
                target = target.cuda()
            else:
                target = torch.tensor([-1])
                target = target.cuda()

            loss_rank = nn.functional.margin_ranking_loss(propsal1, propsal2, target=target, margin=0)
            rankloss += loss_rank
    if i == 3:
        for j in [0, 1, 2, 4]:
            propsal1 = output[i]
            propsal2 = output[j]
            propsal_lab1 = labels[i]
            propsal_lab2 = labels[j]
            if propsal_lab1 > propsal_lab2:
                target = torch.ones(1)
                target = target.cuda()
            else:
                target = torch.tensor([-1])
                target = target.cuda()

            loss_rank = nn.functional.margin_ranking_loss(propsal1, propsal2, target=target, margin=0)
            rankloss += loss_rank
    if i == 4:
        for j in [0, 1, 2, 3]:
            propsal1 = output[i]
            propsal2 = output[j]
            propsal_lab1 = labels[i]
            propsal_lab2 = labels[j]
            if propsal_lab1 > propsal_lab2:
                target = torch.ones(1)
                target = target.cuda()
            else:
                target = torch.tensor([-1])
                target = target.cuda()

            loss_rank = nn.functional.margin_ranking_loss(propsal1, propsal2, target=target, margin=0)
            rankloss += loss_rank


import torch
from dataloader import prepare_dataset
from model import AnticipationModel
from options import parser

opts = parser.parse_args()
train_set, test_set = prepare_dataset(opts)
model = AnticipationModel(opts)

with open(model.log_path, "w") as log_file:

    for epoch in range(1, opts.epochs + 1):

        model.reset_stats()
        model.net.train()
        model.net.set_mode("VARIATIONAL")

        for _, op in train_set:

            hidden_state = model.init_op("train")

            for batch, (data, target_reg, target_cls) in enumerate(op):

                data, target_reg, target_cls = (
                    data.cuda(),
                    target_reg.cuda(),
                    target_cls.cuda(),
                )

                output_reg, output_cls, hidden_state = model.forward(data, hidden_state)
                loss = model.compute_loss(
                    output_reg, output_cls, target_reg, target_cls
                )
                model.backward(loss, batch)

                if model.optimized:
                    h, c = hidden_state
                    hidden_state = (h.detach(), c.detach())

                model.update_stats(
                    loss.item(),
                    output_reg.detach(),
                    output_cls.detach(),
                    target_reg.detach(),
                    target_cls.detach(),
                    mode="train",
                )

            if not model.optimized:
                model.optimizer.step()

        with torch.no_grad():

            # model.net.eval()
            model.net.set_mode("VARIATIONAL")

            if epoch % opts.sample_freq == 0:

                for ID, op in test_set:

                    samples = model.sample_op_predictions(op, opts.num_samples)
                    model.save_samples(samples, ID, epoch)

            # model.net.eval()
            model.net.set_mode("DETERMINISTIC")

            for ID, op in test_set:

                hidden_state = model.init_op("test")

                for data, target_reg, target_cls in op:

                    data, target_reg, target_cls = (
                        data.cuda(),
                        target_reg.cuda(),
                        target_cls.cuda(),
                    )

                    output_reg, output_cls, hidden_state = model.forward(
                        data, hidden_state
                    )
                    loss = model.compute_loss(
                        output_reg, output_cls, target_reg, target_cls
                    )

                    model.update_stats(
                        loss.item(),
                        output_reg.detach(),
                        output_cls.detach(),
                        target_reg.detach(),
                        target_cls.detach(),
                        mode="test",
                    )

        model.summary(log_file, epoch)

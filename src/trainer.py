import torch
from tqdm.auto import tqdm

from .writer import Writer
from .utils import get_logger, MetricTracker


class Trainer:
    """
    Trainer class
    """
    def __init__(self,
                 model_16, model_32,
                 tokenizer, optimizer,
                 config,
                 data_loader, val_data_loader=None,
                 len_epoch: int = None) -> None:
        self.config = config

        self.tokenizer = tokenizer
        self.model_16 = model_16
        self.model_32 = model_32
        self.optimizer = optimizer

        trainer_config = config['trainer']
        self.start_epoch = 1
        self.epochs = trainer_config['epochs']
        self.save_every = trainer_config['save_every']
        self.checkpoint_dir = trainer_config['save_dir']
        self.len_epoch = trainer_config['len_epoch'] or len(data_loader)
        self.len_valid = trainer_config['len_valid'] or len(val_data_loader)

        self.log_dir = self.checkpoint_dir + "/log/" + trainer_config['project_name']
        self.logger = get_logger('trainer', config['trainer']['verbosity'])
        self.writer = Writer(
            self.log_dir, self.logger, trainer_config['writer'], trainer_config['project_name'], config
        )
        self.train_metrics = MetricTracker(
            'loss', 'grad norm', writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            'loss', writer=self.writer
        )
        self.log_step = 10

        self.data_key = config['data']['train']['dataset']['key']
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.do_validation = val_data_loader is not None

        self.accumulate_n = trainer_config['accumulate_n']
        self.batch_size = data_loader.batch_size
        self.valid_batch_size = val_data_loader.batch_size

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info('Saving model on keyboard interrupt')
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information
            log = {'epoch': epoch}
            log.update(result)

            # print logged information
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> dict:
        self.model_16.train()
        self.model_32.train()

        self.train_metrics.reset()
        self.writer.add_scalar('epoch', epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.data_loader, desc='train', total=self.len_epoch),
            start=1
        ):
            try:
                self._train_iteration(batch, batch_idx, epoch)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self.logger.warning('OOM on batch. Skipping batch.')
                    for p in self.model_16.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if batch_idx >= self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        print('Max memory allocated: {:.2f}GB'.format(torch.cuda.max_memory_allocated() / 2 ** 30))
        return log

    def _train_iteration(self, batch, batch_num: int, epoch: int):
        batch = self.tokenizer(
            # batch[self.data_key],
            batch,
            padding=True, truncation=True,
            # max_length=1024,
            # pad_to_multiple_of=256,
            return_tensors='pt'
        )
        batch.to('cuda:0')

        loss = self.model_16(**batch, labels=batch['input_ids'], use_cache=False).loss
        loss = loss / self.accumulate_n
        loss.backward()

        batch.to('cpu')

        self.writer.set_step((epoch - 1) * self.len_epoch + batch_num)
        # self.train_metrics.update('loss', loss.item())
        # self.train_metrics.update('grad norm', self._get_grad_norm())
        grad_norm = self._get_grad_norm()

        if batch_num % self.accumulate_n == 0:
            self._copy_grad()
            self._remove_grad()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self._copy_param()

        if batch_num % self.log_step == 0:
            self._log_scalars(metric_name='loss', val=loss.item())
            self._log_scalars(metric_name='grad norm', val=grad_norm)
            # self._log_scalars(metric_tracker=self.train_metrics)

    def _valid_epoch(self, epoch):
        self.model_16.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(self.val_data_loader, desc='validation', total=self.len_valid),
                start=1
            ):
                batch = self.tokenizer(
                    # batch[self.data_key],
                    batch,
                    padding=True, truncation=True,
                    # max_length=1024,
                    # pad_to_multiple_of=256,
                    return_tensors='pt'
                )
                batch.to('cuda:0')
                loss = self.model_16(**batch, labels=batch['input_ids'], use_cache=False).loss

                batch.to('cpu')

                self.valid_metrics.update('loss', loss.item(), n=self.valid_batch_size)

                if batch_idx >= self.len_valid:
                    break

        self.writer.set_step(epoch * self.len_epoch, 'valid')
        self._log_scalars(metric_tracker=self.valid_metrics)
        # self._log_predictions() TODO

        return self.valid_metrics.result()

    def _log_scalars(self, metric_name: str = None, val: float = None, metric_tracker: MetricTracker = None):
        if self.writer is None:
            return
        if metric_tracker is None:
            self.writer.add_scalar(f'{metric_name}', val)
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f'{metric_name}', metric_tracker.avg(metric_name))

    def _remove_grad(self):
        for p in self.model_16.parameters():
            if p.grad is not None:
                del p.grad
        torch.cuda.empty_cache()

    def _copy_grad(self):
        for p_32, p_16 in zip(self.model_32.parameters(), self.model_16.parameters()):
            p_32.grad = p_16.grad.to('cpu', dtype=torch.float32)

    @torch.no_grad()
    def _copy_param(self):
        for p_32, p_16 in zip(self.model_32.parameters(), self.model_16.parameters()):
            p_16.copy_(p_32.to('cuda', dtype=torch.float16))  # TODO

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        """
        state = {
            'state_dict': self.model_32.state_dict()
        }

        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info('Saving checkpoint: {} ...'.format(filename))

    def _get_grad_norm(self, norm_type=2):
        parameters = self.model_16.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

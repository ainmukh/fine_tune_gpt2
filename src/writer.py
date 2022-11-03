import importlib
from datetime import datetime


class Writer:
    def __init__(self, log_dir, logger, name, project_name, config):
        self.writer = None
        self.selected_module = ""
        self.text_table = None
        modules = ['wandb'] if name == 'wandb' else ["torch.utils.tensorboard", "tensorboardX"]

        # Retrieve vizualization writer.
        if name is not None:
            succeeded = False
            for module in modules:
                try:
                    self.writer = importlib.import_module(module)
                    if name == 'tendorboard':
                        self.writer = self.writer.SummaryWriter(str(log_dir))
                    succeeded = True
                    if name == 'wandb':
                        self.writer.init(project=project_name, config=config)
                    self.selected_module = module
                    break
                except ImportError:
                    succeeded = False

                if not succeeded:
                    instruction = [
                        "TensorboardX with 'pip install tensorboardx'",
                        "WandB with 'pip install wandb"
                    ][name == 'wandb']
                    message = (
                        f"Warning: visualization {name} is configured to use, but currently not installed on "
                        "this machine. Please install"
                        f"{instruction}"
                        ", upgrade PyTorch to "
                        "version >= 1.1 to use this module or turn 'writer' option in the 'config.json' file to 'None'."
                    )
                    logger.warning(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
            "add_audio"
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            if self.selected_module != 'wandb' and name != "add_audio":
                add_data = getattr(self.writer, name, None)

                def wrapper(tag, data, *args, **kwargs):
                    if add_data is not None:
                        # add mode(train/valid) tag
                        if name not in self.tag_mode_exceptions:
                            tag = "{}/{}".format(tag, self.mode)
                        if name == 'add_text':
                            add_data(tag, data)
                        else:
                            add_data(tag, data, self.step, *args, **kwargs)

            else:
                add_data = getattr(self.writer, 'log')

                def wrapper(tag, data, *args, **kwargs):
                    if add_data is not None:
                        if name not in self.tag_mode_exceptions:
                            tag = "{}/{}".format(tag, self.mode)
                    if name == 'add_histogram':
                        add_data({tag: self.writer.Histogram(data.cpu().detach().numpy())}, step=self.step)
                    elif name == 'add_image':
                        add_data({tag: self.writer.Image(data, caption=kwargs["caption"])}, step=self.step)
                    elif name == "add_audio":
                        to_log = []
                        for i, sample in enumerate(data):
                            audio = self.writer.Audio(
                                sample.cpu().detach().numpy(),
                                caption=kwargs["caption"][i],
                                sample_rate=kwargs["sample_rate"]
                            )
                            to_log.append(audio)
                        add_data({tag: to_log}, step=self.step)
                    else:
                        add_data({tag: data}, step=self.step)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        self.selected_module, name
                    )
                )
            return attr

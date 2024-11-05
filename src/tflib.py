import tensorflow as tf

class Checkpoint:
    """
    Enhanced wrapper for TensorFlow's `tf.train.Checkpoint` to simplify checkpoint 
    saving and restoring with additional functionality.
    
    Attributes:
        checkpoint (tf.train.Checkpoint): Checkpoint instance for managing model state.
        manager (tf.train.CheckpointManager): Manages multiple checkpoints and handles 
                                             their storage and retrieval.
    """

    def __init__(self, 
                 checkpoint_kwargs,  # Dict of objects to be saved in the checkpoint
                 directory,  # Directory where checkpoints are stored
                 max_to_keep=5,  # Maximum number of checkpoints to keep
                 keep_checkpoint_every_n_hours=None):  # Interval to keep checkpoints regardless of max_to_keep
        """
        Initializes the Checkpoint with a checkpoint and checkpoint manager.

        Args:
            checkpoint_kwargs (dict): Keyword arguments for `tf.train.Checkpoint`, specifying objects to be tracked.
            directory (str): Directory to store checkpoints.
            max_to_keep (int): Maximum number of checkpoints to keep.
            keep_checkpoint_every_n_hours (float): Interval in hours to save an additional checkpoint.
        """
        # Set up the checkpoint and checkpoint manager
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        """
        Restores the latest checkpoint or a specified checkpoint.

        Args:
            save_path (str, optional): Path to a specific checkpoint. If None, restores the latest checkpoint.

        Returns:
            A status object for running assertions or initializing the checkpoint.
        """
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        """
        Saves a checkpoint, either at a specified location or with a checkpoint number.

        Args:
            file_prefix_or_checkpoint_number (str or int, optional): Prefix or checkpoint number for saving.
            session (tf.Session, optional): TensorFlow session, only needed for specific cases.

        Returns:
            str: Path to the saved checkpoint.
        """
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        """
        Allows access to checkpoint and manager attributes directly from the `Checkpoint` instance.

        Args:
            attr (str): Attribute name to retrieve from checkpoint or manager.

        Returns:
            Attribute from checkpoint or manager if found; raises AttributeError otherwise.
        """
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

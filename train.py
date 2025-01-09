import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter

from config import get_weights_file_path, get_config
import warnings

from tqdm import tqdm

# Generator function to yield all sentences in the dataset for a specific language
# This helps to iterate through the dataset and extract sentences for a particular language
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# Function to get or build a tokenizer for a specific language
# If a saved tokenizer exists, it loads it; otherwise, it trains a new tokenizer from the dataset
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not tokenizer_path.exists():  # Check if the tokenizer file exists
        # Initialize a WordLevel tokenizer and set it to handle unknown tokens
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()  # Split tokens by whitespace
        # Define special tokens and minimum frequency for training
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # Train the tokenizer using sentences from the dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # Save the trained tokenizer to a file
        tokenizer.save(str(tokenizer_path))
    else:
        # Load the pre-trained tokenizer from the file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

# Function to load the dataset and split it into training and validation sets
# Also builds tokenizers for source and target languages
def get_ds(config):
    # Load the dataset using Hugging Face's datasets library
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    # Build tokenizers for both source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Determine the size of the training and validation datasets (90% train, 10% validation)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # Split the dataset into training and validation subsets
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # Create instances of the custom BilingualDataset class for training and validation
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Calculate the maximum sentence lengths for source and target languages
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    # Print the maximum lengths found for source and target sentences
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # Create DataLoader instances for the training and validation datasets
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    # Return the DataLoader instances and the tokenizers for further use
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    # Build a transformer model with source and target vocabulary sizes, sequence length, and model dimension
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Define the device to be used for training (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    # Create the model folder if it doesn't exist to save model checkpoints
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Load the dataset and split it into training and validation dataloaders
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Initialize the transformer model and move it to the appropriate device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Initialize TensorBoard writer to log training details for visualization
    writer = SummaryWriter(config['experiment_name'])
    
    # Define the optimizer with Adam and a small epsilon for numerical stability
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # Initialize epoch and global step for tracking training progress
    initial_epoch = 0
    global_step = 0
    
    # Load a pre-trained model if the preload option is set
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        # Load the model state and optimizer state
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1  # Continue from the next epoch
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    # Define the loss function, using CrossEntropyLoss and ignoring the padding index
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # The rest of the training loop would go here, handling epochs, forward passes, backpropagation, etc.
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) #(b, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)   #(b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)   #(b, 1, seq_len, seq_len)
            
            
            #Run the tensors through the transformers
            encoder_output = model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(b, seq_len, d_model)
            proj_output = model.project(decoder_output) #(b, seq_len, tgt_vocab_size)
            
            label = batch['label'].to(device) # (b, seq_len)
            # (b, seq_len, tgt_vocab_size) --> (b * seq_len, tgt_voca_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(f"loss :" f"{loss.item():6.3f}")
            
            #Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            #Backpropagate
            loss.backward()
            
            #Update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        #Save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
    

    
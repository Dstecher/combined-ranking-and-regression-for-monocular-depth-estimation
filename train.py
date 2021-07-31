import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function, ranking_loss_function, root_mean_squared_error
from utils import predict, save_images, load_test_data
from model import create_model
from data import get_nyu_train_test_data
from callbacks import get_nyu_callbacks
from LRScheduler import CyclicLR

from keras.optimizers import Adam

# Adjusted file compared to Alhashim et al.
# Original code can be found at https://github.com/ialhashim/DenseDepth/blob/master/train.py

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='nyu', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--maxlr', type=float, default=0.00357, help='Learning rate max value for scheduler')
parser.add_argument('--lambda_val', type=float, default=0.5, help='Weighting factor for regression share')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
# Number of sample pairs currently not supported as argument here! This value has to be adjusted manually in loss.py!!

args = parser.parse_args()

# Change this path accordingly
outputPath = './models/' #./models/

# Inform about multi-gpu training (only single-GPU training was used here)
if args.gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')


# Create the model
model = create_model( existing=args.checkpoint )

# Data loaders
if args.data == 'nyu': train_generator, test_generator = get_nyu_train_test_data( args.bs )

# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

 # (optional steps)
if True:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+__file__, 'w') as training_script: training_script.write(training_script_content)

    # Generate model plot
    #plot_model(model, to_file=runPath+'/model_plot.svg', show_shapes=True, show_layer_names=True)

    # Save model summary to file
    from contextlib import redirect_stdout
    with open(runPath+'/model_summary.txt', 'w') as f:
        with redirect_stdout(f): model.summary()

# Multi-gpu setup:
basemodel = model
#if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus) # There was a fault with this function in TF2.0, therefore only single-GPU training was used

# Combined loss function implemented here for easier access to lambda argument
def combined_loss_function(y_true, y_pred):

    l = args.lambda_val

    reg_loss = depth_loss_function(y_true, y_pred)
    
    rank_loss = ranking_loss_function(y_true, y_pred)

    if rank_loss == None:
        # In case the ranking loss is None (deprecated), only use regression loss
        print("[INFO] Rank loss is None!")
        return reg_loss
    else:
        return l * reg_loss + (1 - l) * rank_loss

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')

model.compile(loss=combined_loss_function, optimizer=optimizer, metrics=[root_mean_squared_error], run_eagerly=True)

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'nyu': callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)

# Cyclic LR Schedule information
clr_step_size = 25344 # 2 * 12672 (steps_per_epoch) default value
base_lr = args.lr
max_lr = args.maxlr
mode = 'triangular'
lr_schedule = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=mode)
callbacks.append( lr_schedule ) # Append to callbacks

# Start training
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True)

# Save the final trained model:
basemodel.save(runPath + '/model.h5')

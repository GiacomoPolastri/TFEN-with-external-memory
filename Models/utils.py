from matplotlib import pyplot as plt
import torch

# Get the number to save the training 
def get_number_model():
    file = open('./Models/number_last_model.txt', 'r')
    lines = file.readlines()
    number = lines[0]
    print('Train model number: {}'.format(number))
    file.close()
        
    return number

# Change the number for the next training
def change_number_model(number):
    file = open('./Models/number_last_model.txt', 'w')
    numb = int(number) + 1
    file.write(str(numb))
    print ('Numero di training cambiato da {} a {}'.format(number, numb))
    file.close()

    return numb

# Save the results
def save_model(epochs, model, optimizer, criterion, best_acc, n):

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'best_acc': best_acc,
                }, './Train/' + n + '.pth')

# Save the plots
def save_plots(train_acc, valid_acc, train_loss, valid_loss, n):
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./Outputs/accuracy' + n + '.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Outputs/loss' + n + '.png')

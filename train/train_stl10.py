import torch
import torchvision
import argparse
import logging
import os
import architectures.color_96x96 as architectures

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    datasetDirectory,
    outputDirectory,
    batchSize,
    architecture,
    dropoutRatio,
    learningRate,
    weightDecay,
    numberOfEpochs,
    useCuda,
    saveAccuracyChampion
):
    logging.info(f"train_stl10.main(); useCuda = {useCuda}; architecture = {architecture}")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    device = 'cpu'
    if useCuda:
        device = 'cuda'

    # Load the dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=0.3, std=0.5)]
    )
    stl10_dataset = torchvision.datasets.STL10(root=datasetDirectory, split='train',
                                       transform=transform)
    train_dataset, validation_dataset = torch.utils.data.random_split(stl10_dataset, [0.8, 0.2])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batchSize)

    # Create the neural network
    neural_net = None
    if architecture == 'ThreeConv2Lin_64_128_256_1024':
        neural_net = architectures.ThreeConv2Lin(
            number_of_channels1=64,
            number_of_channels2=128,
            number_of_channels3=256,
            linear1_size=1024,
            dropout_ratio=dropoutRatio
        )
    elif architecture == 'ThreeConv2Lin_64_128_256_256':
        neural_net = architectures.ThreeConv2Lin(
            number_of_channels1=64,
            number_of_channels2=128,
            number_of_channels3=256,
            linear1_size=256,
            dropout_ratio=dropoutRatio
        )
    elif architecture == 'ThreeRes1Lin_64':
        neural_net = architectures.ThreeRes1Lin(64)
    elif architecture == 'Resx6Linx1_64':
        neural_net = architectures.Resx6Linx1(64)
    elif architecture == 'Resx3x3Linx1_64_32':
        neural_net = architectures.Resx3x3Linx1(
            residual_channels123=64,
            residual_channels456=32
        )
    elif architecture == 'ResTrios_64_32_16':
        neural_net = architectures.ResTrios(
            residual_channels_list=[64, 32, 16]
        )
    elif architecture == 'ResTrios_128_64_32':
        neural_net = architectures.ResTrios(
            residual_channels_list=[128, 64, 32]
        )
    elif architecture == 'ResTrios_64_32':
        neural_net = architectures.ResTrios(
            residual_channels_list=[64, 32]
        )
    elif architecture == 'ResTrios_32_32_32':
        neural_net = architectures.ResTrios(
            residual_channels_list=[32, 32, 32]
        )
    else:
        raise NotImplementedError(f"Not implemented architecture '{architecture}'")
    neural_net.to(device)

    # Training parameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learningRate, weight_decay=weightDecay)

    lowest_validation_loss = 1.0e9
    highest_accuracy = 0.0
    championship_criterion = "validation loss"
    if saveAccuracyChampion:
        championship_criterion = "accuracy"

    # Training monitoring file
    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w') as epoch_loss_file:
        epoch_loss_file.write(f"epoch,training_loss,validation_loss,accuracy,is_champion\n")
        for epoch in range(numberOfEpochs):
            neural_net.train()
            running_loss = 0.0
            number_of_batches = 0
            for train_input_tsr, train_target_output_tsr in train_dataloader:
                train_input_tsr = train_input_tsr.to(device)
                train_target_output_tsr = train_target_output_tsr.to(device)
                # Set the parameter gradients to zero before every batch
                neural_net.zero_grad()
                # Pass the input tensor through the neural network
                output_tsr = neural_net(train_input_tsr)
                # Compute the loss, i.e., the error function we want to minimize
                loss = criterion(output_tsr, train_target_output_tsr)
                # Back-propagate the loss function to compute the gradient of the loss with respect
                # to every trainable parameter in the neural network
                loss.backward()
                # Perturb every trainable parameter by a small quantity, in the direction of the steepest gradient descent
                optimizer.step()

                running_loss += loss.item()
                number_of_batches += 1
                if number_of_batches % 10 == 1:
                    print('.', flush=True, end='')
            training_loss = running_loss/number_of_batches

            # Evaluate with the validation dataset
            neural_net.eval()
            validation_running_loss = 0.0
            number_of_batches = 0
            number_of_correct_predictions = 0
            number_of_predictions = 0
            for validation_input_tsr, validation_target_output_tsr in validation_dataloader:
                validation_input_tsr = validation_input_tsr.to(device)
                validation_target_output_tsr = validation_target_output_tsr.to(device)
                # Pass the input tensor through the neural network
                validation_output_tsr = neural_net(validation_input_tsr)
                # Compute the validation loss
                loss = criterion(validation_output_tsr, validation_target_output_tsr)
                validation_running_loss += loss.item()
                number_of_correct_predictions += numberOfCorrectPredictions(validation_output_tsr, validation_target_output_tsr)
                number_of_predictions += validation_input_tsr.shape[0]
                number_of_batches += 1
            validation_loss = validation_running_loss/number_of_batches
            accuracy = number_of_correct_predictions/number_of_predictions

            is_champion = False
            if validation_loss < lowest_validation_loss:
                lowest_validation_loss = validation_loss
                if not saveAccuracyChampion:
                    is_champion = True

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                if saveAccuracyChampion:
                    is_champion = True

            if is_champion:
                neural_net_filepath = os.path.join(outputDirectory, f"{architecture}.pth")
                torch.save(neural_net.state_dict(), neural_net_filepath)


            logging.info(f" **** Epoch {epoch} ****\ntraining_loss = {training_loss}; validation_loss = {validation_loss}; accuracy = {accuracy}")
            if is_champion:
                logging.info(f" + + + + Champion for {championship_criterion}! Saving {neural_net_filepath} + + + +")

            epoch_loss_file.write(f"{epoch},{training_loss},{validation_loss},{accuracy},{is_champion}\n")

def numberOfCorrectPredictions(predictions_tsr, target_class_tsr):
    return sum(torch.argmax(predictions_tsr, dim=1) == target_class_tsr).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasetDirectory', help="The directory containing the 'stl10_binary' directory")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train_stl10'", default='./output_train_stl10')
    parser.add_argument('--batchSize', help="The batch size. Default: 64", type=int, default=64)
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'ThreeConv2Lin_64_128_256_1024'", default='ThreeConv2Lin_64_128_256_1024')
    parser.add_argument('--dropoutRatio', help="The dropout ratio. Default: 0.5", type=float, default=0.5)
    parser.add_argument('--learningRate', help="The learning rate. Default: 0.001", type=float, default=0.001)
    parser.add_argument('--weightDecay', help="The weight decay. Default: 0.00001", type=float, default=0.00001)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 50", type=int, default=50)
    parser.add_argument('--useCpu', help="Use CPU, even if there is a GPU", action='store_true')
    parser.add_argument('--saveAccuracyChampion', help="Save the accuracy champion, instead of the validation loss champion", action='store_true')
    args = parser.parse_args()

    useCuda = torch.cuda.is_available() and not args.useCpu

    main(
        args.datasetDirectory,
        args.outputDirectory,
        args.batchSize,
        args.architecture,
        args.dropoutRatio,
        args.learningRate,
        args.weightDecay,
        args.numberOfEpochs,
        useCuda,
        args.saveAccuracyChampion
    )
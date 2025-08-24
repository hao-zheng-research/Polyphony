import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence

# Replace the import to use our improved model
from improved_dual_hand_model_v3_fixed import DualHandASDiffusionModel
from dual_hand_dataset import DualHandVideoFeatureDataset, get_dual_hand_data_dicts
from enhanced_coordination_loss import EnhancedDualHandCoordinationLoss
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


class SimpleImprovedDualHandTrainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device, args=None):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.args = args

        # Use the improved model (now with architectural coordination)
        self.model = DualHandASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Simple Improved Dual Hand Model Size: ', sum(p.numel() for p in self.model.parameters()))
        
        # Initialize enhanced coordination loss system
        self.enhanced_coordination_loss = EnhancedDualHandCoordinationLoss(self.device, self.num_classes)
        
        print("🎯 Using Simple Improved Architecture with:")
        print("   - Lightweight cross-hand feature fusion")
        print("   - Enhanced coordination loss system")
        print("   - Strong coordination mismatch penalties")

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, lh_label_dir, rh_label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        # Enhanced optimizer with lower learning rate for coordination stability
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate*0.9, weight_decay=weight_decay, eps=1e-8)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate*0.1
        )
        
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1
        best_avg_acc = 0.0
        best_coordination_score = 0.0

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    try:
                        saved_state = torch.load(os.path.join(result_dir, 'latest.pt'), weights_only=False)
                        self.model.load_state_dict(saved_state['model'], strict=False)
                        optimizer.load_state_dict(saved_state['optimizer'])
                        restore_epoch = saved_state['epoch']
                        step = saved_state['step']
                        if 'best_coordination_score' in saved_state:
                            best_coordination_score = saved_state['best_coordination_score']
                        print(f"📁 Resumed from epoch {restore_epoch}")
                    except Exception as e:
                        print(f"⚠️  Could not load checkpoint: {e}")

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        print(f"\n🏋️ Starting Simple Improved Training for {num_epochs} epochs")
        
        for epoch in range(restore_epoch+1, num_epochs):
            # Update coordination loss with current epoch for adaptive weights
            self.enhanced_coordination_loss.update_epoch(epoch)
            
            self.model.train()
            epoch_running_loss = 0
            epoch_coordination_loss = 0
            epoch_enhanced_coord_loss = 0
            
            # Progress tracking
            epoch_progress = tqdm(train_train_loader, desc=f"Epoch {epoch}")
            
            for batch_idx, data in enumerate(epoch_progress):

                (lh_feature, lh_label, lh_boundary, 
                 rh_feature, rh_label, rh_boundary, video) = data
                
                lh_feature, lh_label, lh_boundary = lh_feature.to(device), lh_label.to(device), lh_boundary.to(device)
                rh_feature, rh_label, rh_boundary = rh_feature.to(device), rh_label.to(device), rh_boundary.to(device)
                
                # Enhanced diffusion loss with architectural coordination
                loss_dict = self.model.get_training_loss(
                    lh_feature, rh_feature,
                    lh_event_gt=F.one_hot(lh_label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    rh_event_gt=F.one_hot(rh_label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    lh_boundary_gt=lh_boundary,
                    rh_boundary_gt=rh_boundary,
                    encoder_ce_criterion=ce_criterion,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                total_loss = 0
                coord_loss_total = torch.tensor(0.0, device=device)
                
                # Standard losses
                for k, v in loss_dict.items():
                    if k in ['feature_coordination', 'temporal_coordination']:
                        coord_loss_total += v * 0.5  # Reduced weight for new simpler coordination
                    else:
                        base_key = k.replace('lh_', '').replace('rh_', '')
                        if base_key in loss_weights:
                            total_loss += loss_weights[base_key] * v

                # Get encoder outputs for enhanced coordination loss
                lh_encoder_out, _ = self.model.left_hand_model.encoder(lh_feature, get_features=True)
                rh_encoder_out, _ = self.model.right_hand_model.encoder(rh_feature, get_features=True)
                
                # Compute all enhanced coordination losses
                enhanced_losses = self.enhanced_coordination_loss.compute_all_losses(
                    lh_encoder_out, rh_encoder_out, lh_label, rh_label
                )
                
                # Add both coordination losses
                enhanced_coord_total = enhanced_losses['total_enhanced_coordination']
                total_loss += coord_loss_total + enhanced_coord_total
                
                epoch_coordination_loss += coord_loss_total.item()
                epoch_enhanced_coord_loss += enhanced_coord_total.item()

                # Update progress bar
                epoch_progress.set_postfix({
                    'Total Loss': f'{total_loss.item():.4f}',
                    'Arch+Loss': f'{(coord_loss_total + enhanced_coord_total).item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })

                if result_dir and step % 10 == 0:
                    # Log standard losses
                    for k, v in loss_dict.items():
                        if k.startswith('coord_'):
                            logger.add_scalar(f'Train-{k}', v.item(), step)
                        else:
                            base_key = k.replace('lh_', '').replace('rh_', '')
                            if base_key in loss_weights:
                                logger.add_scalar(f'Train-{k}', loss_weights[base_key] * v.item() / batch_size, step)
                    
                    # Log enhanced coordination losses
                    for loss_name, loss_value in enhanced_losses.items():
                        if isinstance(loss_value, torch.Tensor):
                            logger.add_scalar(f'Train-Enhanced-{loss_name}', loss_value.item(), step)
                    
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)
                    logger.add_scalar('Train-Arch-Coordination', coord_loss_total.item(), step)
                    logger.add_scalar('Train-Enhanced-Coordination', enhanced_coord_total.item(), step)
                    logger.add_scalar('Learning-Rate', scheduler.get_last_lr()[0], step)

                total_loss /= batch_size
                total_loss.backward()
                
                # Enhanced gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            # Update learning rate
            scheduler.step()
            
            epoch_running_loss /= len(train_train_dataset)
            epoch_coordination_loss /= len(train_train_dataset)
            epoch_enhanced_coord_loss /= len(train_train_dataset)

            print(f'\nEpoch {epoch} Summary:')
            print(f'  Total Loss: {epoch_running_loss:.6f}')
            print(f'  Architectural Coordination Loss: {epoch_coordination_loss:.6f}')
            print(f'  Enhanced Coordination Loss: {epoch_enhanced_coord_loss:.6f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
            if result_dir:
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'best_coordination_score': best_coordination_score
                }

            if epoch % log_freq == 0:
                if result_dir:
                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # Test both hands
                for mode in ['decoder-agg']: 
                    test_result_dict = self.test(
                        test_test_dataset, mode, device, lh_label_dir, rh_label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k, v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)
                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    # Calculate coordination score
                    coordination_score = (test_result_dict.get('Avg_Acc', 0) + 
                                        test_result_dict.get('Avg_F1@50', 0)) / 2.0
                    
                    # Print detailed results
                    print(f'\n📊 Epoch {epoch} - {mode} Test Results:')
                    for k, v in test_result_dict.items():
                        print(f'  {k}: {v:.4f}')
                    print(f'  Coordination Score: {coordination_score:.4f}')

                    # Save best models
                    current_avg_acc = test_result_dict.get('Avg_Acc', 0.0)
                    if current_avg_acc > best_avg_acc:
                        best_avg_acc = current_avg_acc
                        if result_dir:
                            torch.save(self.model.state_dict(), f'{result_dir}/best_accuracy_model.model')
                            print(f'🎯 New best accuracy model saved! Avg_Acc: {best_avg_acc:.4f}')
                    
                    if coordination_score > best_coordination_score:
                        best_coordination_score = coordination_score
                        if result_dir:
                            torch.save(self.model.state_dict(), f'{result_dir}/best_coordination_model.model')
                            print(f'🤝 New best coordination model saved! Score: {best_coordination_score:.4f}')

                    if log_train_results:
                        train_result_dict = self.test(
                            train_test_dataset, mode, device, lh_label_dir, rh_label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k, v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        print(f'\n📈 Epoch {epoch} - {mode} Train Results:')
                        for k, v in train_result_dict.items():
                            print(f'  {k}: {v:.4f}')
                        
        if result_dir:
            logger.close()
            
        print(f"\n🎉 Training completed!")
        print(f"   Best Accuracy: {best_avg_acc:.4f}")
        print(f"   Best Coordination Score: {best_coordination_score:.4f}")

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            (lh_feature, lh_label, lh_boundary,
             rh_feature, rh_label, rh_boundary, video) = test_dataset[video_idx]

            if mode == 'encoder':
                lh_output, rh_output = self.model.get_encoders_output(
                    torch.stack([f.to(device) for f in lh_feature]),
                    torch.stack([f.to(device) for f in rh_feature])
                )
                lh_output = [F.softmax(lh_output, 1).cpu()]
                rh_output = [F.softmax(rh_output, 1).cpu()]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-agg':
                lh_outputs = []
                rh_outputs = []
                for i in range(len(lh_feature)):
                    lh_out, rh_out = self.model.ddim_sample_both_hands(
                        lh_feature[i].to(device), rh_feature[i].to(device), seed)
                    lh_outputs.append(lh_out.cpu())
                    rh_outputs.append(rh_out.cpu())
                
                lh_output = lh_outputs
                rh_output = rh_outputs
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-noagg':  # temporal aug must be true
                lh_out, rh_out = self.model.ddim_sample_both_hands(
                    lh_feature[len(lh_feature)//2].to(device), 
                    rh_feature[len(rh_feature)//2].to(device), seed)
                lh_output = [lh_out.cpu()]
                rh_output = [rh_out.cpu()]
                left_offset = self.sample_rate // 2
                right_offset = 0

            # Process left hand output
            assert(lh_output[0].shape[0] == 1)
            min_len = min([i.shape[2] for i in lh_output])
            lh_output = [i[:,:,:min_len] for i in lh_output]
            lh_output = torch.cat(lh_output, 0).mean(0).numpy()

            # Process right hand output  
            assert(rh_output[0].shape[0] == 1)
            min_len = min([i.shape[2] for i in rh_output])
            rh_output = [i[:,:,:min_len] for i in rh_output]
            rh_output = torch.cat(rh_output, 0).mean(0).numpy()

            # Apply postprocessing to both hands
            def apply_postprocessing(output, label):
                if self.postprocess['type'] == 'median':
                    smoothed_output = np.zeros_like(output)
                    for c in range(output.shape[0]):
                        smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                    output = smoothed_output / smoothed_output.sum(0, keepdims=True)

                output = np.argmax(output, 0)
                output = restore_full_sequence(output, 
                    full_len=label.shape[-1], 
                    left_offset=left_offset, 
                    right_offset=right_offset, 
                    sample_rate=self.sample_rate
                )

                if self.postprocess['type'] == 'mode':
                    output = mode_filter(output, self.postprocess['value'])

                if self.postprocess['type'] == 'purge':
                    trans, starts, ends = get_labels_start_end_time(output)
                    for e in range(0, len(trans)):
                        duration = ends[e] - starts[e]
                        if duration <= self.postprocess['value']:
                            if e == 0:
                                output[starts[e]:ends[e]] = trans[e+1]
                            elif e == len(trans) - 1:
                                output[starts[e]:ends[e]] = trans[e-1]
                            else:
                                mid = starts[e] + duration // 2
                                output[starts[e]:mid] = trans[e-1]
                                output[mid:ends[e]] = trans[e+1]
                return output

            lh_output = apply_postprocessing(lh_output, lh_label)
            rh_output = apply_postprocessing(rh_output, rh_label)
            
            lh_label = lh_label.squeeze(0).cpu().numpy()
            rh_label = rh_label.squeeze(0).cpu().numpy()

            assert(lh_output.shape == lh_label.shape)
            assert(rh_output.shape == rh_label.shape)
            
            return video, lh_output, lh_label, rh_output, rh_label

    def test(self, test_dataset, mode, device, lh_label_dir, rh_label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset)), desc="Testing"):
                
                video, lh_pred, lh_label, rh_pred, rh_label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                lh_pred = [self.event_list[int(i)] for i in lh_pred]
                rh_pred = [self.event_list[int(i)] for i in rh_pred]
                
                # Save left hand predictions
                if not os.path.exists(os.path.join(result_dir, 'lh_prediction')):
                    os.makedirs(os.path.join(result_dir, 'lh_prediction'))

                lh_file_name = os.path.join(result_dir, 'lh_prediction', f'{video}.txt')
                with open(lh_file_name, 'w') as file_ptr:
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(lh_pred))

                # Save right hand predictions
                if not os.path.exists(os.path.join(result_dir, 'rh_prediction')):
                    os.makedirs(os.path.join(result_dir, 'rh_prediction'))

                rh_file_name = os.path.join(result_dir, 'rh_prediction', f'{video}.txt')
                with open(rh_file_name, 'w') as file_ptr:
                    file_ptr.write('### Frame level recognition: ###\n')
                    file_ptr.write(' '.join(rh_pred))

        # Evaluate both hands
        lh_acc, lh_edit, lh_f1s = func_eval(
            lh_label_dir, os.path.join(result_dir, 'lh_prediction'), test_dataset.video_list)
        
        rh_acc, rh_edit, rh_f1s = func_eval(
            rh_label_dir, os.path.join(result_dir, 'rh_prediction'), test_dataset.video_list)

        result_dict = {
            'LH_Acc': lh_acc,
            'LH_Edit': lh_edit,
            'LH_F1@10': lh_f1s[0],
            'LH_F1@25': lh_f1s[1],
            'LH_F1@50': lh_f1s[2],
            'RH_Acc': rh_acc,
            'RH_Edit': rh_edit,
            'RH_F1@10': rh_f1s[0],
            'RH_F1@25': rh_f1s[1],
            'RH_F1@50': rh_f1s[2],
            'Avg_Acc': (lh_acc + rh_acc) / 2,
            'Avg_Edit': (lh_edit + rh_edit) / 2,
            'Avg_F1@10': (lh_f1s[0] + rh_f1s[0]) / 2,
            'Avg_F1@25': (lh_f1s[1] + rh_f1s[1]) / 2,
            'Avg_F1@50': (lh_f1s[2] + rh_f1s[2]) / 2,
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--simple_improved', action='store_true', default=True,
                       help='Use simple improved architecture with both architectural and loss coordination')
    args = parser.parse_args()
    view_id = "View0"

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print("🚀 SIMPLE IMPROVED DUAL-HAND TRAINING")
    print("=" * 60)
    print(args.config)
    print("Simple improved architecture enabled")
    print("Key improvements:")
    print("  - Lightweight cross-hand feature fusion (architectural)")
    print("  - Enhanced coordination loss system (loss-based)")
    print("  - Strong coordination mismatch penalties")
    print("  - Backward compatible with existing configs")
    print("=" * 60)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    mapping_file = os.path.join(root_data_dir, dataset_name, 'task_mapping.txt')
    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    # Load dual hand data
    (lh_train_data_dict, lh_test_data_dict, 
     rh_train_data_dict, rh_test_data_dict) = get_dual_hand_data_dicts(
        root_data_dir, dataset_name, split_id, event_list,
        sample_rate, temporal_aug, boundary_smooth, view_id
    )

    # Wraps both hand datasets and ensures video alignment
    train_train_dataset = DualHandVideoFeatureDataset(
        lh_train_data_dict, rh_train_data_dict, num_classes, mode='train')
    train_test_dataset = DualHandVideoFeatureDataset(
        lh_train_data_dict, rh_train_data_dict, num_classes, mode='test')
    test_test_dataset = DualHandVideoFeatureDataset(
        lh_test_data_dict, rh_test_data_dict, num_classes, mode='test')

    trainer = SimpleImprovedDualHandTrainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        args=args
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Get label directories for evaluation
    lh_label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/lh_pt')
    rh_label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth/View0/rh_pt')

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        lh_label_dir=lh_label_dir, rh_label_dir=rh_label_dir,
        result_dir=os.path.join(result_dir, naming + "_simple_improved"), 
        log_freq=log_freq, log_train_results=log_train_results
    ) 
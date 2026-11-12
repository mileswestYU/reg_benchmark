import os.path as osp
import argparse
import time
import torch
import numpy as np
import os
import torch.utils.data
from config import config
from vision3d.engine import Engine
from vision3d.utils.torch_utils import to_cuda

from model import create_model
from dataset import ROBI_test_data_loader
from loss import Evaluator
from vision3d.utils.point_cloud_utils import  apply_transform



eps = 1e-8

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, required=True, default=0)


def run_one_epoch(
        engine,
        epoch,
        data_loader,
        model,
        evaluator,
        training=True
):
    if training:
        model.train()
        if engine.distributed:
            data_loader.sampler.set_epoch(epoch)
    else:
        model.eval()
    num_iter_per_epoch = len(data_loader)
    inlier_ratio_num=0
    mean_pre=0
    mean_recall=0
    count=0
    for i, data_dict in enumerate(data_loader):
        print(i)
        data_dict = to_cuda(data_dict)
        with torch.no_grad():
            output_dict = model(data_dict)
            estimated_transforms_gt=data_dict['transform']
            all_ref_corr_points = output_dict['all_ref_corr_points']
            all_src_corr_points = output_dict['all_src_corr_points']

            # EXPORT_MIRETR_CORRESPONDENCES_V1
            # ROBI中：src是物体模型点云，ref是场景点云。
            model_corr_points = all_src_corr_points.detach().cpu().numpy().astype(np.float32)
            scene_corr_points = all_ref_corr_points.detach().cpu().numpy().astype(np.float32)
            corr_scores = output_dict['all_corr_scores'].detach().cpu().numpy().astype(np.float32)

            estimated_transforms = output_dict['estimated_transforms'].detach().cpu().numpy().astype(np.float32)
            gt_transforms = data_dict['transform'].detach().cpu().numpy().astype(np.float32)

            frame_info = data_dict['frag_id1']
            object_name = str(frame_info[0])

            scene_value = frame_info[1]
            if torch.is_tensor(scene_value):
                scene_value = scene_value.detach().cpu().item()
            scene_id = int(scene_value)

            view_value = frame_info[2]
            if torch.is_tensor(view_value):
                view_value = view_value.detach().cpu().item()
            view_name = str(view_value)

            symmetry = str(data_dict['frag_id0'])

            # 每行：[model_x, model_y, model_z, scene_x, scene_y, scene_z]
            corr_pairs = np.concatenate(
                [model_corr_points, scene_corr_points],
                axis=1
            ).astype(np.float32)

            sample_output_dir = os.path.join(
                config.features_dir,
                object_name,
                'Scene_{}'.format(scene_id)
            )
            os.makedirs(sample_output_dir, exist_ok=True)

            safe_view_name = view_name.replace('/', '_').replace('\\\\', '_')
            output_file = os.path.join(
                sample_output_dir,
                '{}.npz'.format(safe_view_name)
            )

            np.savez_compressed(
                output_file,
                object_name=np.asarray(object_name),
                scene_id=np.asarray(scene_id, dtype=np.int32),
                view_name=np.asarray(view_name),
                symmetry=np.asarray(symmetry),

                # 推荐接入自己算法的字段
                corr_pairs=corr_pairs,
                corr_scores=corr_scores,

                # 分开存储的对应点
                model_corr_points=model_corr_points,
                scene_corr_points=scene_corr_points,
                src_corr_points=model_corr_points,
                ref_corr_points=scene_corr_points,

                # MIRETR和GT位姿
                estimated_transforms=estimated_transforms,
                gt_transforms=gt_transforms
            )

            print(
                '[EXPORT] {} Scene_{} {}: {} correspondences -> {}'.format(
                    object_name,
                    scene_id,
                    view_name,
                    corr_pairs.shape[0],
                    output_file
                ),
                flush=True
            )
            result_dict = evaluator(output_dict, data_dict)
            mean_pre+=result_dict['precision']
            mean_recall+=result_dict['recall']
            print('precision',100*mean_pre/(count+1))
            print('recall',100*mean_recall/(count+1))    
            corr_tensor=torch.cat((all_src_corr_points, all_ref_corr_points), dim=1)
            align_src_points = apply_transform(corr_tensor[:,:3].unsqueeze(0), estimated_transforms_gt)
            rmse = torch.linalg.norm(align_src_points - corr_tensor[:,3:].unsqueeze(0), dim=-1)<(0.005)
            inlier_ratio=(rmse.float().sum(0)>0)
            inlier_ratio=inlier_ratio.float().sum()/len(inlier_ratio)
            if len(all_src_corr_points)>1:
                inlier_ratio_num+=inlier_ratio
            print('IR',100*inlier_ratio_num/(count+1)) 
            count+=1            
    print('precision',100*mean_pre/count)
    print('recall',100*mean_recall/count)    
    recall=100*mean_recall/count
    precision=100*mean_pre/count
    print("f1",2 * (precision ) * (recall) / ((recall  +precision)))

    message = 'precision: {:.3f}, '.format(100*mean_pre/count) + \
                            'recall: {:.3f}, '.format(100*mean_recall/count) + \
                                'inlier_ratio: {:.3f}, '.format(100*inlier_ratio_num/(count))
    engine.logger.critical(message)



def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        engine.args.test_epoch=39

        start_time = time.time()

        test_loader,neighborhood_limits = ROBI_test_data_loader(engine, config)
        loading_time = time.time() - start_time

        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model)

        snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(engine.args.test_epoch))
        engine.load_snapshot(snapshot)

        start_time = time.time()
        run_one_epoch(
                engine, engine.args.test_epoch, test_loader, model, evaluator, training=False
            )

        loading_time = time.time() - start_time
        message = ' test_one_epoch: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)


if __name__ == '__main__':
    main()

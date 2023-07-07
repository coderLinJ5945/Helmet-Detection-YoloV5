"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam  各种类型的文件
        imgsz=640,  # inference size (pixels ) 推断大小：像素
        conf_thres=0.25,  # confidence threshold 置信度阈值
        iou_thres=0.45,  # NMS IOU threshold NMS IOU阈值
        max_det=1000,  # maximum detections per image 每张图片的最大检测次数：1000次
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu 使用的设备类型：该模型只支持GPU
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos 是否保存推理后的结果图像
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    # save inference images 保存推理结果图像
    save_img = not nosave and not source.endswith('.txt')
    # isnumeric() 方法检测字符串是否只由数字组成。这种方法是只针对unicode对象。
    # endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回 True，否则返回 False。
    # 以下代码判断source是否为数字，或者以.txt结尾，或者以rtsp://、rtmp://、http://、https://开头，如果是则认为是视频流
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 推理结果图像保存的文件夹 Directories  判断文件夹是否存在，在文件名后面加上数字，如果文件名已经存在，那么数字加1
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 是否保存labels文件
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    # 配置开启日志
    set_logging()
    # 配置选择 cpu or cuda（0,1,2,3 为使用gpu的核数）
    device = select_device(device)
    # 半精度方法只支持CUDA模式
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
    # Load model 加载模型：如果weights是list类型，则返回weights[0]，否则返回weights
    w = weights[0] if isinstance(weights, list) else weights
    # 设置模型的推理类型：如果以 .pt 结尾，那么classify = True，否则 onnx = False
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    # stride：模型中的最大步长 names：模型中的类别
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        # 加载pt模型权重文件，返回模型
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride 模型中的最大步长

        # names的输出结果为：['person', 'hat'] 是模型中训练设置的类别
        # 如果模型中有module属性则返回 model.module.names 否则返回 model.names 这里更像是做兼容判断？
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # 将所有浮点参数和缓冲区强制转换为``half``数据类型，half 数据类型只支持CUDA
        if half:
            # 将模型转换为半精度模型
            model.half()  # to FP16
        if classify:  # second-stage classifier
            # 加载分类器
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    # 检测图片的设置大小，将输入的图片大小转换为32的倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader 加载数据 如果是视频流则使用LoadStreams，否则使用LoadImages
    if webcam:
        # web网络摄像头视频流数据
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        # 图片文件数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # 执行模型推理过程
    if pt and device.type != 'cpu':
        # ？ todo
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # 用于计算推理的耗时时间
    t0 = time.time()

    print(type(dataset))
    # path是图片的路径，img是处理后的图片的数据，im0s是原始图片数据，vid_cap是视频流数据
    for path, img, im0s, vid_cap in dataset:
        if pt:
            # from_numpy()将numpy数据转换为tensor数据,用于GPU计算

            img = torch.from_numpy(img).to(device)
            # tensor([[[114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          ...,
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114]],
            #
            #         [[114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          ...,
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114]],
            #
            #         [[114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          ...,
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114],
            #          [114, 114, 114,  ..., 114, 114, 114]]], dtype=torch.uint8)
            # 如果使用半精度模式，将img数据转换为半精度数据，否则转换为float数据
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        # img /= 255.0 的作用是将图片数据转换为0.0 - 1.0的数据 ，归一化处理？
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # tensor([[[0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          ...,
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706]],
        #
        #         [[0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          ...,
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706]],
        #
        #         [[0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          ...,
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706],
        #          [0.44706, 0.44706, 0.44706,  ..., 0.44706, 0.44706, 0.44706]]])
        if len(img.shape) == 3:
            img = img[None]  # img[None]的作用是在img的第0维度增加一个维度，变成4维数据，用于后续的模型输入

        # Inference
        t1 = time_sync()
        if pt:
            # increment_path的作用是在save_dir路径下创建一个新的文件夹，文件夹名称为path的文件名
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # model的作用是将img数据输入到模型中，得到预测结果， pred是预测结果
            pred = model(img, augment=augment, visualize=visualize)[0]
            # tensor([[[3.89326e+00, 4.07307e+00, 9.61285e+00,  ..., 9.52162e-08, 8.96032e-01, 7.59697e-02],
            #          [1.40034e+01, 4.30870e+00, 1.97850e+01,  ..., 7.95964e-09, 8.14778e-01, 1.19378e-01],
            #          [2.13431e+01, 4.01182e+00, 2.24153e+01,  ..., 8.52310e-09, 8.14820e-01, 1.28592e-01],
            #          ...,
            #          [5.52513e+02, 6.24010e+02, 2.16757e+02,  ..., 2.67440e-06, 6.68891e-01, 1.52486e-01],
            #          [5.84867e+02, 6.21800e+02, 1.60438e+02,  ..., 1.79149e-06, 8.23164e-01, 7.75056e-02],
            #          [6.17497e+02, 6.04189e+02, 1.29241e+02,  ..., 2.11730e-06, 7.77198e-01, 9.38333e-02]]])
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS non_max_suppression的作用是对预测结果进行非极大值抑制，得到最终的预测结果
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # 1个预测结果 30.07462 为左上角x坐标，34.68237为左上角y坐标，486.09238为右下角x坐标，490.19916为右下角y坐标，0.91738为置信度，1.00000为类别
        # [tensor([[130.07462,  34.68237, 486.09238, 490.19916,   0.91738,   1.00000]])]
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        # 处理预测结果： i是图片的索引，det是图片的预测结果
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                # p为图片的路径，s为图片的尺寸，im0为图片的数据，frame为图片的帧数
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            # Path方法的作用是将字符串转换为Path对象
            p = Path(p)  # to Path
            # save_path为图片的保存路径，txt_path为标签的保存路径 'runs/detect/exp58/595a84814d0e4c9a91c0ea7e30214ac7.jpeg'
            save_path = str(save_dir / p.name)  # img.jpg
            # txt_path：'runs/detect/exp58/labels/595a84814d0e4c9a91c0ea7e30214ac7'
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            # s为图片的尺寸，s += '%gx%g ' % img.shape[2:]的作用是将图片的尺寸拼接到s字符串中 '640x640 ' %gx%g的作用？
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn： tensor([664, 633, 664, 633]) 664为图片的宽度，633为图片的高度
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc是图片的副本，如果save_crop为True，则imc为im0的副本，否则imc为im0 ，im0为图片的元数据
            imc = im0.copy() if save_crop else im0  # for save_crop

            if len(det):  # 如果预测结果不为空

                # [:, :4]表示取所有行的前4列，即取预测结果的坐标，scale_coords的作用是将预测结果的坐标从img_size尺寸转换为im0尺寸
                # shape[2:]：tensor([640, 640]) 640为图片的宽度，640为图片的高度

                # round 将张量的每个元素舍入为最近的整数
                # 处理前的det: tensor([[130.07462,  34.68237, 486.09238, 490.19916,   0.91738,   1.00000]]) 尺寸是1x6
                # 处理后的det： tensor([[135.00000,  20.00000, 504.00000, 493.00000,   0.91738,   1.00000]]) 尺寸是1x6
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results 打印预测结果
                # [:, -1].unique()的作用是取最后一列的所有值，即取预测结果的类别
                for c in det[:, -1].unique():
                    # n为预测结果的数量，names[int(c)]为预测结果的类别
                    # n的结果为1，names[int(c)]的结果为 'hat'
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results 将预测结果写入文件
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file   如果save_txt为True，则将预测结果写入txt文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh [0.4811747074127197, 0.40521326661109924, 0.5557228922843933, 0.7472353577613831]
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format (tensor(1.), 0.4811747074127197, 0.40521326661109924, 0.5557228922843933, 0.7472353577613831)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 如果save_img为True，则将预测的图片保存到save_path路径下
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class c c为预测结果的类别 1
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 'hat 0.92'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness) # 将预测结果画在原图片上
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    # 启动命令demo：
    # python helmet_yolov5/detect.py
    # --source test/595a84814d0e4c9a91c0ea7e30214ac7.jpeg
    # --weights model_files/helmet.pt --device cpu
    # --imgsz 640
    # --view-img

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # 2. 检查验证环境：requirements.txt中的软件包版本要求是否匹配
    check_requirements(exclude=('tensorboard', 'thop'))
    # 3. 运行模型预测推理
    run(**vars(opt))
    # import torch
    # print(torch.cuda.is_available())
    # print(torch.backends.cudnn.is_available())
    # print(torch.cuda_version)
    # print(torch.backends.cudnn.version())

if __name__ == "__main__":
    # 1. opt 启动main方法参数，有学习借鉴意义
    opt = parse_opt()
    main(opt)

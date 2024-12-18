import os

from TableStructureRec.lineless_table_rec import LinelessTableRecognition
from TableStructureRec.lineless_table_rec.utils_table_recover import (
    format_html,
    plot_rec_box_with_logic_info,
    plot_rec_box,
)
from TableStructureRec.table_cls import TableCls
from TableStructureRec.wired_table_rec import WiredTableRecognition
from RapidOCR.python.rapidocr_onnxruntime import RapidOCR



def table_ocr(img_path):
    lineless_engine = LinelessTableRecognition()
    wired_engine = WiredTableRecognition()
    # 默认小yolo模型(0.1s)，可切换为精度更高yolox(0.25s),更快的qanything(0.07s)模型
    table_cls = TableCls(model_type="yolox")  # TableCls(model_type="yolox"),TableCls(model_type="q")

    cls, elasp = table_cls(img_path)
    if cls == "wired":
        table_engine = wired_engine
    else:
        table_engine = lineless_engine

    ocr_engine = RapidOCR(
        cls_label_list=["0", "0"],
        det_model_dir="RapidOCR/python/rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer.onnx",
        rec_model_dir="RapidOCR/python/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx")
    ocr_res, _ = ocr_engine(img_path)

    print(ocr_res)

    html, elasp, polygons, logic_points, ocr_res = table_engine(
        img_path, ocr_result=ocr_res)
    print(f"elasp: {elasp}")
    return html, elasp, polygons, logic_points, ocr_res


# 使用其他ocr模型
# ocr_engine =RapidOCR(det_model_dir="xxx/det_server_infer.onnx",rec_model_dir="xxx/rec_server_infer.onnx")
# ocr_res, _ = ocr_engine(img_path)
# html, elasp, polygons, logic_points, ocr_res = table_engine(img_path, ocr_result=ocr_res)


def out_info(output_dir, html, ocr_res, logic_points, polygons, img_path):
    import os

    img_filename = os.path.basename(img_path)

    file_less_ext, ext = os.path.splitext(img_filename)
    table_filename = file_less_ext.replace("-tran", "-table") + ".html"

    complete_html = format_html(html)
    output_dir_path = os.path.dirname(f"{output_dir}/{table_filename}")
    print(f"output_dir_path: {output_dir_path}")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
    with open(f"{output_dir}/{table_filename}", "w", encoding="utf-8") as file:
        file.write(complete_html)

    # 可视化表格识别框 + 逻辑行列信息
    table_rec_box_filename = file_less_ext.replace("-tran", "-table_rec_box") + ".jpg"
    plot_rec_box_with_logic_info(
        img_path, f"{output_dir}/{table_rec_box_filename}", logic_points, polygons
    )

    table_ocr_box = file_less_ext.replace("-tran", "-table_ocr_box") + ".jpg"
    # 可视化 ocr 识别框
    plot_rec_box(img_path, f"{output_dir}/{table_ocr_box}", ocr_res)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("use ./" + sys.argv[0] + "input.png")
        exit(1)

    img_path = sys.argv[1]
    html, elasp, polygons, logic_points, ocr_res = table_ocr(img_path)

    out_info(
        html=html,
        output_dir=f"outputs",
        ocr_res=ocr_res,
        logic_points=logic_points,
        polygons=polygons,
        img_path=img_path,
    )

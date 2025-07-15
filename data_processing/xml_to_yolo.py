import os
import xml.etree.ElementTree as ET

class YOLOConverter:
    """YOLO格式转换器"""
    CLASS_MAP = {
        'holothurian': 0,
        'echinus': 1,
        'scallop': 2,
        'starfish': 3
    }

    @staticmethod
    def convert_file(xml_path, txt_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in YOLOConverter.CLASS_MAP:
                    continue
                    
                box = obj.find('bndbox')
                coords = [int(box.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                x_center = (coords[0] + coords[2]) / 2 / img_w
                y_center = (coords[1] + coords[3]) / 2 / img_h
                width = (coords[2] - coords[0]) / img_w
                height = (coords[3] - coords[1]) / img_h
                
                line = f"{YOLOConverter.CLASS_MAP[name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)

def batch_convert(xml_dir, output_dir):
    """批量转换XML到YOLO格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_dir):
        if not xml_file.lower().endswith('.xml'):
            continue
            
        xml_path = os.path.join(xml_dir, xml_file)
        txt_path = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')
        
        try:
            YOLOConverter.convert_file(xml_path, txt_path)
            print(f"转换成功: {xml_file} -> {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"转换失败 {xml_file}: {str(e)}")

if __name__ == "__main__":
    batch_convert(
        xml_dir=r"E:\BaiduNetdiskDownload\train\box",
        output_dir=r"E:\BaiduNetdiskDownload\train\labels"
    )
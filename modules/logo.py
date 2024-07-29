from termcolor import colored
import shutil
import pyfiglet

def create_logo():
    # 创建主要的Logo内容
    xray_text = "Xray"
    tag_text = "TAG"
    
    figlet = pyfiglet.Figlet(font='slant')
    xray_logo = figlet.renderText(xray_text)
    tag_logo = figlet.renderText(tag_text)

    # 将Xray和TAG部分分别渲染为彩色
    xray_lines = xray_logo.split('\n')
    tag_lines = tag_logo.split('\n')

    # 合并Xray和TAG部分
    combined_logo_lines = []
    for xray_line, tag_line in zip(xray_lines, tag_lines):
        combined_logo_lines.append(colored(xray_line, 'cyan') + colored(tag_line, 'yellow'))

    combined_logo = "\n".join(combined_logo_lines)
    
    return combined_logo
def create_furnace():
    # 炉身主体部分
    furnace_body = [
        "           一      ____________     三",
        "           跑     /            \\    轮",
        "           即    /      炼      \\   就",
        "           通   |       丹       |  成",
        "           无   |       炉       |  新",
        "           罢   \\               /   嗖",
        "           搁    \\_____________/    踏",
    ]
    
    # 炉的装饰部分
    furnace_decor = [
        "                //\\    ||     /\\\\    ",
        "               //  \\___||____/  \\\\   ",
    ]
    
    # # 炉脚
    # furnace_legs = [
    #     "   ( (   ) (   ) )  ",
    #     "   ) ) (   ) ) ( (  ",
    # ]
    
    # 添加火焰效果
    smoke = [
        "                    ( (  (  (     ",
        "                     )  )  )      ",
        "                    (  (  (       ",
    ]

    # 添加烟雾效果
    fire = [
        "                    ~~~~~~~~      ",
        "                  ~~~~~~~~~~~~    ",
    ]

    # 使用颜色渲染
    colored_smoke = [colored(line, 'white') for line in smoke]
    colored_furnace_body = [colored(line, 'yellow') for line in furnace_body]
    colored_furnace_decor = [colored(line, 'yellow') for line in furnace_decor]
    # colored_furnace_legs = [colored(line, 'yellow') for line in furnace_legs]
    colored_fire = [colored(line, 'red') for line in fire]

    # 组合所有部分
    combined_art = "\n".join(colored_smoke + colored_furnace_body + colored_furnace_decor + colored_fire)

    return combined_art

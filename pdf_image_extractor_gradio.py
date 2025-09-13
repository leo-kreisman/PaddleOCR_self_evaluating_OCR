import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import sys
import gradio as gr
import time
import io

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (A mesma de antes) ---
# Esta função contém toda a lógica de extração e melhoria.
def extrair_e_melhorar_pdf(pdf_path, output_dir, initial_dpi=400):
    """
    Extrai cada página de um PDF, renderiza em alta resolução e aplica
    super-resolução com Real-ESRGAN para máxima qualidade.
    """
    print("-> Configurando o modelo de IA Real-ESRGAN...")
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        use_gpu = 'cuda' in sys.modules.get('torch.version', '')
        half_precision = use_gpu
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model, tile=0, tile_pad=10, pre_pad=0, half=half_precision
        )
        print("-> Modelo configurado com sucesso.")
    except Exception as e:
        error_msg = f"ERRO: Falha ao carregar o modelo RealESRGAN. Verifique sua conexão ou dependências. Erro: {e}"
        print(error_msg)
        raise gr.Error(error_msg)

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error_msg = f"ERRO: Não foi possível abrir o arquivo PDF. Verifique se o arquivo não está corrompido. Erro: {e}"
        print(error_msg)
        raise gr.Error(error_msg)

    image_paths = []
    for page_num in range(len(doc)):
        page_index = page_num + 1
        print(f"\n--- Processando Página {page_index} de {len(doc)} ---")
        page = doc.load_page(page_num)

        print(f"  (1/3) Renderizando página em {initial_dpi} DPI...")
        mat = fitz.Matrix(initial_dpi / 72, initial_dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        print(f"  (2/3) Aplicando super-resolução com IA...")
        try:
            np_image = np.array(img_pil, dtype=np.uint8)
            output_image_np, _ = upsampler.enhance(np_image, outscale=4)
            output_image_pil = Image.fromarray(output_image_np)
        except Exception as e:
            print(f"  AVISO: Falha ao aplicar IA na página {page_index}. Usando a imagem de alta DPI renderizada. Erro: {e}")
            output_image_pil = img_pil # Usa a imagem de alta DPI se a IA falhar

        output_filename = f"pagina_{page_index:04d}_melhorada.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"  (3/3) Salvando imagem final em '{output_path}'...")
        output_image_pil.save(output_path, "PNG", dpi=(initial_dpi * 4, initial_dpi * 4))
        image_paths.append(output_path)
    
    return image_paths

# --- FUNÇÃO WRAPPER PARA A INTERFACE GRADIO ---
# Esta função conecta a lógica de processamento com os componentes da interface.
def processar_pdf_gradio(pdf_file_obj):
    """
    Função chamada pela interface Gradio. Recebe o objeto de arquivo, processa,
    e retorna os caminhos das imagens para a galeria.
    """
    if pdf_file_obj is None:
        raise gr.Error("Por favor, envie um arquivo PDF!")

    # Usa o caminho temporário do arquivo enviado pelo Gradio
    input_pdf_path = pdf_file_obj.name
    
    # Cria um diretório de saída único para esta execução para evitar conflitos
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"output_{timestamp}"

    status_message = f"Processando... As imagens serão salvas em '{output_dir}'."
    print(status_message)
    
    # Chama a função principal de processamento
    try:
        final_image_paths = extrair_e_melhorar_pdf(input_pdf_path, output_dir)
        status_message = f"**Sucesso!** {len(final_image_paths)} páginas foram processadas e salvas no diretório `{output_dir}` no seu computador."
        return final_image_paths, status_message
    except Exception as e:
        # Se ocorrer um erro durante o processamento, ele será exibido na interface
        # A exceção já foi tratada com gr.Error dentro da função principal
        return None, str(e)


# --- CRIAÇÃO DA INTERFACE GRADIO ---
if __name__ == "__main__":
    print("Iniciando a interface Gradio...")
    
    # Define a interface com título, descrição e componentes de entrada/saída
    iface = gr.Interface(
        fn=processar_pdf_gradio,
        inputs=gr.File(label="Selecione o arquivo PDF", file_types=[".pdf"]),
        outputs=[
            gr.Gallery(label="Imagens Melhoradas", show_label=True, elem_id="gallery"),
            gr.Markdown(label="Status")
        ],
        title="Extrator e Otimizador de Imagens de PDF com IA",
        description="Arraste e solte um arquivo PDF abaixo. O sistema irá extrair cada página, converter para uma imagem de alta qualidade e usar a IA (Real-ESRGAN) para quadriplicar a resolução. As imagens resultantes serão exibidas abaixo e salvas em um diretório 'output' no seu computador.",
        allow_flagging="never",
        examples=[["exemplo.pdf"]] # Você pode colocar um PDF de exemplo na pasta
    )

    # Inicia o servidor web local
    iface.launch()
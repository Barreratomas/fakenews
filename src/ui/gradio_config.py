import gradio as gr

CSS = """
    .gradio-container {
        max-width: 100% !important;
        padding: 20px !important;
    }
    .row {
        flex-wrap: wrap !important;
    }
    @media (max-width: 768px) {
        .column {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    
    /* Animación Radar Clásico */
    .radar-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        margin: 20px auto;
        max-width: 300px;
        position: relative;
        overflow: hidden;
    }
    
    .radar-circle {
        width: 80px;
        height: 80px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        position: relative;
        margin-bottom: 15px;
    }
    
    .radar-pulse {
        position: absolute;
        width: 100%;
        height: 100%;
        border: 2px solid rgba(255, 255, 255, 0.8);
        border-radius: 50%;
        animation: radar-pulse 2s infinite;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    
    .radar-text {
        color: white;
        font-size: 16px;
        font-weight: 500;
        opacity: 0.9;
    }
    
    @keyframes radar-pulse {
        0% {
            transform: translate(-50%, -50%) scale(0.5);
            opacity: 1;
        }
        100% {
            transform: translate(-50%, -50%) scale(2);
            opacity: 0;
        }
    }
    
    @media (max-width: 768px) {
        .radar-container {
            max-width: 250px;
            padding: 20px;
        }
        .radar-circle {
            width: 60px;
            height: 60px;
        }
        .radar-text {
            font-size: 14px;
        }
    }

    /* Ocultar footer de Gradio */
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    """

def create_gradio_interface(predict_fn):
    """Crea y configura la interfaz de Gradio con el layout simplificado y responsive."""
    
    with gr.Blocks(title="Detector de Fake News") as demo:
        gr.Markdown(
            """
            # Detector de Fake News IA
            Ingrese el texto de una noticia o una URL para analizar su veracidad.
            """
        )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=400):
                input_type = gr.Radio(
                    ["Texto", "URL"], 
                    label="Tipo de entrada", 
                    value="Texto"
                )
                
                text_input = gr.Textbox(
                    label="Texto de la noticia", 
                    placeholder="Pega aquí el contenido completo de la noticia...", 
                    lines=8,
                    visible=True,
                    max_lines=15
                )
                
                url_input = gr.Textbox(
                    label="URL de la noticia", 
                    placeholder="https://ejemplo.com/noticia", 
                    visible=False
                )
                
                # Componente oculto para el Session ID
                session_id_box = gr.Textbox(visible=False)

                btn_analyze = gr.Button("Analizar Noticia", variant="primary", size="lg")
                
                # Contenedor del radar (inicialmente oculto)
                radar_container = gr.HTML(
                    value="""<div class="radar-container" style="display: none;">
                        <div class="radar-circle">
                            <div class="radar-pulse"></div>
                        </div>
                        <div class="radar-text">Analizando noticia...</div>
                    </div>""",
                    visible=True
                )
                
        # Sección de Resultados (Output) - Simplificada y responsive
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=400):
                label_output = gr.HTML(label="Veredicto")
                extracted_output = gr.Textbox(
                    label="Texto Analizado", 
                    lines=10, 
                    interactive=False,
                    max_lines=15
                )

        # Eventos de visibilidad
        def toggle_inputs(choice):
            if choice == "Texto":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
            
        input_type.change(toggle_inputs, input_type, [text_input, url_input], show_progress=False)
        
        # Función wrapper que maneja el radar
        def analyze_with_radar(input_option, text_input, url_input, session_id):
            # 1) Mostrar radar: devolvemos el HTML visible
            yield ("", "", """<div class="radar-container">
                <div class="radar-circle">
                    <div class="radar-pulse"></div>
                </div>
                <div class="radar-text">Analizando noticia...</div>
            </div>""")
            
            # 2) Ejecutar el análisis
            label_result, extracted_result = predict_fn(input_option, text_input, url_input, session_id)
            
            # 3) Ocultar radar: devolvemos los resultados y cadena vacía para el radar
            yield (label_result, extracted_result, "")
        
        # Evento de análisis - Sin animación de progreso
        btn_analyze.click(
            fn=analyze_with_radar,
            inputs=[input_type, text_input, url_input, session_id_box],
            outputs=[label_output, extracted_output, radar_container],
            show_progress=False,
            queue=True
        )
        
        # Cargar Session ID al inicio
        demo.load(
            fn=None,
            inputs=None,
            outputs=session_id_box,
            js="""() => {
                let sessionId = sessionStorage.getItem("custom_session_id");
                if (!sessionId) {
                    sessionId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
                    sessionStorage.setItem("custom_session_id", sessionId);
                }
                return sessionId;
            }"""
        )
        
        return demo
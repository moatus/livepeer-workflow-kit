FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/livepeer:/workspace/livepeer/references/roboflow-inference
ENV WORKFLOWS_PLUGIN_ONLY_CORE=true
ENV WORKFLOWS_PLUGINS=roboflow_livepeer_blocks
ENV DISABLE_VERSION_CHECK=true
ENV METRICS_ENABLED=false
ENV CORE_MODEL_GAZE_ENABLED=false
ENV GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
ENV RN_FORCE_SINK="fakesink sync=true async=false"
ENV RN_FORCE_AUDIO_SINK="fakesink sync=true async=false"
ENV XDG_RUNTIME_DIR=/tmp/xdg-runtime

WORKDIR /workspace/livepeer

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    gir1.2-gst-plugins-bad-1.0 \
    python3 \
    python3-pip \
    python3-venv \
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libgirepository-1.0-1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-nice \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-x \
    gstreamer1.0-gl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY docker/requirements.workbench.txt ./docker/requirements.workbench.txt
COPY roboflow_livepeer_blocks ./roboflow_livepeer_blocks

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    -r docker/requirements.workbench.txt \
    -e .

COPY scripts ./scripts
COPY tests ./tests
COPY references/raspberry_ninja ./references/raspberry_ninja
COPY references/roboflow-inference ./references/roboflow-inference
COPY docker/entrypoint.sh /usr/local/bin/livepeer-poc
RUN chmod +x /usr/local/bin/livepeer-poc

ENTRYPOINT ["livepeer-poc"]
CMD ["help"]

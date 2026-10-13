import os
import time
# Force pure Python protobuf to prevent C++ core dumps on Raspberry Pi ARM64
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, Response, jsonify, render_template_string
import cv2
import numpy as np
import threading
import ai_edge_litert.interpreter as tflite

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. INITIALIZE CAMERA
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------------------------------------------------------------------
# 2. LOAD THE QUANTIZED TFLITE MODEL
# ---------------------------------------------------------------------------
MODEL_PATH = "crop_disease_quantized.tflite"
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

CLASSES = [
    "Cassava Bacterial Blight", 
    "Cassava Brown Streak", 
    "Cassava Green Mottle", 
    "Cassava Mosaic", 
    "Healthy", 
    "Unknown"
]

# Global variables for threading and screen display
current_frame = None
encoded_jpeg = None
lock = threading.Lock()

# Global variables for 5-second on-screen results
latest_result_text = ""
result_clear_time = 0

# ---------------------------------------------------------------------------
# 3. ON-SCREEN DISPLAY & BACKGROUND CAMERA LOOP
# ---------------------------------------------------------------------------
def camera_loop():
    global current_frame, encoded_jpeg, latest_result_text, result_clear_time
    
    # Create a full-screen window for the physical Raspberry Pi screen
    cv2.namedWindow("Crop Scanner Feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Crop Scanner Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.1)
            continue
        
        # Center-crop the 640x480 frame into a perfect square
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Resize to match AI input
        square_frame = cv2.resize(square_frame, (320, 320))

        with lock:
            current_frame = square_frame.copy()
            
        # Draw on-screen diagnostic target frame for the operator's view
        display_frame = square_frame.copy()
        p1, p2 = (32, 32), (288, 288) 
        color = (154, 255, 45) # Neon Green (BGR)
        cv2.rectangle(display_frame, p1, p2, color, 1, lineType=cv2.LINE_AA)
        
        cl = 16 
        cv2.line(display_frame, (32, 32), (32+cl, 32), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (32, 32), (32, 32+cl), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (288, 32), (288-cl, 32), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (288, 32), (288, 32+cl), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (32, 288), (32+cl, 288), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (32, 288), (32, 288-cl), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (288, 288), (288-cl, 288), color, 3, lineType=cv2.LINE_AA)
        cv2.line(display_frame, (288, 288), (288, 288-cl), color, 3, lineType=cv2.LINE_AA)
        
        cv2.putText(display_frame, "ALIGN LEAF IN BOX", (36, 52), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, lineType=cv2.LINE_AA)

        # Draw the scan result if it's within the 5-second window
        if time.time() < result_clear_time:
            # Add a dark semi-transparent banner at the bottom
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 290), (320, 320), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Color logic: Green for Healthy, Red for Diseases
            res_color = (154, 255, 45) if "Healthy" in latest_result_text else (78, 59, 255)
            cv2.putText(display_frame, latest_result_text, (10, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, res_color, 1, cv2.LINE_AA)

        # Save encoded image for the Web UI stream
        with lock:
            ret_encode, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret_encode:
                encoded_jpeg = buffer.tobytes()

        # Render to the physical Raspberry Pi screen
        cv2.imshow("Crop Scanner Feed", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def generate_video():
    # Feeds the Web UI without blocking the main physical screen loop
    while True:
        with lock:
            if encoded_jpeg is None:
                continue
            frame_bytes = encoded_jpeg
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05) # Prevent overloading the network

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------------------------------------------------------
# 4. AI INFERENCE ENDPOINT
# ---------------------------------------------------------------------------
@app.route('/analyze')
def analyze_plant():
    global current_frame, latest_result_text, result_clear_time
    with lock:
        if current_frame is None:
            return jsonify({"status": "error", "message": "No camera frame available"})
        frame_to_analyze = current_frame.copy()

    roi = frame_to_analyze[32:288, 32:288]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_roi, (224, 224), interpolation=cv2.INTER_AREA)
    
    input_data = (resized.astype(np.float32) / 127.5) - 1.0
    
    if input_details[0]['dtype'] == np.uint8:
        scale, zp = input_details[0]['quantization']
        if scale == 0: scale = 1.0
        input_tensor = np.clip(np.round(input_data / scale + zp), 0, 255).astype(np.uint8)
    else:
        input_tensor = np.expand_dims(input_data, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]['index'])[0].astype(np.float32)
    
    if output_details[0]['dtype'] == np.uint8:
        scale, zp = output_details[0]['quantization']
        if scale:
            y = (y - zp) * scale

    if y.min() < 0 or abs(float(y.sum()) - 1.0) > 0.05:
        e = np.exp(y - y.max())
        y = e / e.sum()

    disease_probs = y[:5] 
    if disease_probs.sum() > 0:
        disease_probs = disease_probs / disease_probs.sum() 
    
    pred_idx = np.argmax(disease_probs)
    confidence = float(disease_probs[pred_idx]) * 100.0
    disease_name = CLASSES[pred_idx]

    # UPDATE GLOBAL VARIABLES TO TRIGGER PHYSICAL SCREEN OVERLAY FOR 5 SECS
    latest_result_text = f"{disease_name} ({round(confidence, 1)}%)"
    result_clear_time = time.time() + 5.0

    return jsonify({
        "status": "success",
        "disease": disease_name,
        "confidence": round(confidence, 1)
    })

# ---------------------------------------------------------------------------
# 5. EMBEDDED WEB DASHBOARD
# ---------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Crop Scanner</title>
<style>
:root{
  --cy:#19e3ff; --gr:#2dff9a; --rd:#ff3b4e; --am:#ffb020;
  --bg:#070a12; --panel:#0e1322; --edge:#1b2340; --dim:#6f7c9c;
  --gap:clamp(12px,2.5vmin,24px);
  --dp:min(52dvh,34vw);
  --font:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
html,body{margin:0;width:100%;height:100%;background:var(--bg);color:#e4ecff;font-family:var(--font);touch-action:none;overscroll-behavior:none}
body{background:radial-gradient(900px 600px at 50% 0%,#152040 0%,#070a12 70%); overflow-x:hidden;}
button{font:inherit;color:inherit;border:0;margin:0;padding:0;background:none;cursor:pointer;outline:0;touch-action:none}
.app{position:fixed;inset:0;height:100dvh;display:grid;grid-template-rows:auto 1fr;gap:var(--gap);padding:calc(env(safe-area-inset-top) + var(--gap)) calc(env(safe-area-inset-right) + var(--gap)) calc(env(safe-area-inset-bottom) + var(--gap)) calc(env(safe-area-inset-left) + var(--gap));}
.hdr{display:flex;align-items:center;justify-content:space-between;gap:12px;width:100%;max-width:1400px;margin:0 auto;}
.brand{font-weight:900;letter-spacing:.25em;font-size:clamp(15px,3.4vmin,24px);background:linear-gradient(90deg,#fff,var(--cy));-webkit-background-clip:text;background-clip:text;color:transparent}
.pill{display:flex;align-items:center;gap:8px;padding:.5em 1.1em;border-radius:99px;background:#090c14;border:1px solid var(--edge);font-weight:800;font-size:clamp(10px,2.2vmin,13px);letter-spacing:.15em;color:var(--rd)}
.pill i{width:.8em;height:.8em;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor}
.pill.ok{color:var(--gr)}
.pad{display:grid;gap:var(--gap);width:100%;max-width:1400px;margin:0 auto;align-items:center;justify-items:center;}
.ctl{transition:opacity .15s,filter .15s}
body.offline .ctl{opacity:.3;pointer-events:none;filter:grayscale(.7)}
.tact{background:linear-gradient(180deg,#39405a,#232839 55%,#1a1e2d);box-shadow:0 .5vmin 0 #05070c,0 1.2vmin 2vmin rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.16);transition:transform .08s,box-shadow .08s,color .08s}
.tact.on,.tact:active{transform:translateY(.5vmin);color:var(--cy);box-shadow:0 .1vmin 0 #05070c,0 0 3vmin rgba(25,227,255,.5),inset 0 0 2.5vmin rgba(25,227,255,.28)}
.dpad{position:relative;width:var(--dp);height:var(--dp);max-width:340px;max-height:340px;display:grid;grid-template:repeat(3,1fr)/repeat(3,1fr);padding:3%;border-radius:50%;background:radial-gradient(circle at 50% 35%,#171b29,#0a0c14);box-shadow:inset 0 4px 14px rgba(0,0,0,.8),0 0 0 1px var(--edge)}
.db{border-radius:22%;font-size:calc(var(--dp)*.13);display:flex;align-items:center;justify-content:center;margin:2%}
.up{grid-area:1/2}.lf{grid-area:2/1}.rt{grid-area:2/3}.dn{grid-area:3/2}
.hub{grid-area:2/2;border-radius:50%;background:radial-gradient(#0c0f18,#161a27);margin:8%;box-shadow:inset 0 3px 8px #000;display:flex;align-items:center;justify-content:center}
.hub b{width:35%;height:35%;border-radius:50%;background:#2a3147;box-shadow:0 0 0 3px #10131d;transition:background .15s,box-shadow .15s}
.dpad.live .hub b{background:var(--gr);box-shadow:0 0 14px var(--gr)}
.dpad.locked .db{opacity:.55}
.center{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;width:100%;max-width:380px;}
.cam-feed {width:100%;border-radius:12px;border:2px solid var(--edge);box-shadow:0 4px 14px rgba(0,0,0,.8);aspect-ratio:1;object-fit:cover;}
.btn-analyze {padding:14px 24px;background:linear-gradient(90deg,var(--cy),#007bff);color:#fff;border-radius:8px;font-weight:900;font-size:18px;letter-spacing:.05em;box-shadow:0 4px 10px rgba(25,227,255,.3);width:100%;}
.result-box {font-size:15px;font-weight:bold;color:var(--gr);text-align:center;min-height:48px;display:flex;align-items:center;justify-content:center;background:#090c14;border-radius:8px;width:100%;padding:10px;border:1px solid var(--edge);}
.card{width:100%;padding:16px 20px;border-radius:18px;background:var(--panel);border:1px solid var(--edge);text-align:center}
.lbl{font-size:11px;letter-spacing:.2em;color:var(--dim);font-weight:700}
.dir{font-size:30px;font-weight:900;letter-spacing:.12em;color:var(--cy);text-shadow:0 0 14px rgba(25,227,255,.4);margin:4px 0;}
.bar{height:10px;border-radius:9px;background:#05070c;margin-top:10px;overflow:hidden;box-shadow:inset 0 2px 4px #000}
.bar i{display:block;height:100%;width:30%;border-radius:9px;background:linear-gradient(90deg,var(--gr),var(--cy));transition:width .15s}
.spd{font-size:18px;font-weight:800;margin-top:8px}
.acts{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:var(--gap);width:100%;max-width:340px}
.drive{width:100%;height:60px;border-radius:18px;font-weight:900;letter-spacing:.15em;font-size:16px;}
.drive.active{color:var(--gr);box-shadow:0 .1vmin 0 #05070c,0 0 3vmin rgba(45,255,154,.5),inset 0 0 2.5vmin rgba(45,255,154,.25);transform:translateY(.5vmin)}
.stop{width:130px;aspect-ratio:1;border-radius:50%;font-weight:900;letter-spacing:.1em;font-size:20px;color:#fff;background:radial-gradient(circle at 50% 30%,#ff6a78,#d81f35 65%,#8f0f20);box-shadow:0 .8vmin 0 #4a0710,0 1.6vmin 2.4vmin rgba(0,0,0,.6),0 0 3vmin rgba(255,59,78,.4);transition:transform .08s,box-shadow .08s}
.stop:active{transform:translateY(.7vmin);box-shadow:0 .1vmin 0 #4a0710,0 0 4vmin rgba(255,59,78,.8)}
.sp{display:flex;gap:var(--gap);width:100%}
.sp button{flex:1;height:60px;border-radius:18px;font-weight:900;font-size:28px;}

@media (min-width: 850px) {
  .pad{grid-template-columns: 1fr minmax(340px, 420px) 1fr;}
}

@media (max-width: 849px) {
  .app{position:relative; height:auto; min-height:100dvh; overflow-y:auto;}
  :root{--dp:min(70vw,350px)}
  .pad{grid-template-columns:1fr; grid-template-rows:auto auto auto; align-content:start;}
  .center{order:1; margin-bottom:10px;}
  .dpadwrap{order:2; margin-bottom:10px;}
  .acts{order:3; max-width:100%; display:grid; grid-template-columns:1fr auto; align-items:center;}
  .acts .drive{grid-column:1}
  .acts .sp{grid-column:1}
  .acts .stop{grid-column:2; grid-row:1/3}
}
</style>
</head>
<body class="offline">
<div class="app">
  <div class="hdr">
    <div class="brand">CROP SCANNER</div>
    <div class="pill" id="pill"><i></i><span id="ptxt">OFFLINE</span></div>
  </div>
  <div class="pad">
    <div class="dpadwrap ctl" style="display:flex;justify-content:center;width:100%">
      <div class="dpad locked" id="dpad">
        <button class="db tact up" data-c="MOVE_FORWARD">&#9650;</button>
        <button class="db tact lf" data-c="TURN_LEFT">&#9664;</button>
        <div class="hub"><b></b></div>
        <button class="db tact rt" data-c="TURN_RIGHT">&#9654;</button>
        <button class="db tact dn" data-c="MOVE_BACKWARD">&#9660;</button>
      </div>
    </div>
    <div class="center">
      <img src="/video_feed" class="cam-feed" alt="Camera Feed Offline">
      <button class="btn-analyze tact" id="analyzeBtn">SCAN PLANT</button>
      <div id="aiResult" class="result-box">Ready for inspection...</div>
      <div class="card">
        <div class="lbl">DIRECTION</div>
        <div class="dir" id="dir">IDLE</div>
        <div class="bar"><i id="bar"></i></div>
        <div class="spd"><span class="lbl">SPEED </span><span id="spd">40</span>%</div>
      </div>
    </div>
    <div class="acts ctl">
      <button class="drive tact" id="drive">DRIVE OFF</button>
      <div class="sp">
        <button class="tact" id="sdn">&minus;</button>
        <button class="tact" id="sup">+</button>
      </div>
      <button class="stop" id="stop">STOP</button>
    </div>
  </div>
</div>
<script>
(function(){
  var $=function(i){return document.getElementById(i)};
  var ws=null,online=false,drive=false,held=null,heldPtr=null,timer=null;

  function send(c){ if(ws&&ws.readyState===1) ws.send(c); }

  function setOnline(v){
    online=v; document.body.classList.toggle('offline',!v);
    $('pill').classList.toggle('ok',v);$('ptxt').textContent=v?'ROBOT LINKED':'ROBOT OFFLINE';
    if(!v){ release(); }
  }

  function apply(s){
    drive=!!s.drive;
    $('drive').textContent=drive?'DRIVE ON':'DRIVE OFF';
    $('drive').classList.toggle('active',drive);$('dpad').classList.toggle('locked',!drive);
    $('dpad').classList.toggle('live',drive);$('spd').textContent=s.speed; $('bar').style.width=s.speed+'\%';$('dir').textContent=s.direction;
  }

  function connect(){
    try{ ws=new WebSocket('ws://192.168.4.1:81/'); }catch(e){ setTimeout(connect,1000); return; }
    ws.onopen=function(){ setOnline(true); send('PING'); };
    ws.onclose=function(){ setOnline(false); setTimeout(connect,1000); };
    ws.onerror=function(){ try{ws.close()}catch(e){} };
    ws.onmessage=function(e){ try{ apply(JSON.parse(e.data)); }catch(x){} };
  }

  function press(cmd,el,id){
    if(!drive) return;
    held=cmd; heldPtr=id;
    document.querySelectorAll('.db').forEach(function(b){b.classList.remove('on')});
    if(el) el.classList.add('on');
    send(cmd);
    clearInterval(timer);
    timer=setInterval(function(){ send(cmd); }, 150);
  }

  function release(id){
    if(held===null) return;
    if(id!==undefined && id!==heldPtr) return;
    clearInterval(timer); timer=null; held=null; heldPtr=null;
    document.querySelectorAll('.db').forEach(function(b){b.classList.remove('on')});
    send('STOP');
  }

  document.querySelectorAll('.db').forEach(function(btn){
    btn.addEventListener('pointerdown',function(e){
      e.preventDefault(); try{btn.setPointerCapture(e.pointerId)}catch(x){}
      press(btn.dataset.c,btn,e.pointerId);
    });
    ['pointerup','pointercancel','lostpointercapture'].forEach(function(ev){
      btn.addEventListener(ev,function(e){ release(e.pointerId); });
    });
    btn.addEventListener('contextmenu',function(e){e.preventDefault()});
  });

  $('drive').addEventListener('pointerdown',function(e){ e.preventDefault(); release(); send(drive?'DRIVE_OFF':'DRIVE_ON'); });$('stop').addEventListener('pointerdown',function(e){ e.preventDefault(); release(); send('STOP'); });
  $('sup').addEventListener('pointerdown',function(e){ e.preventDefault(); send('SPEED_UP'); });$('sdn').addEventListener('pointerdown',function(e){ e.preventDefault(); send('SPEED_DOWN'); });

  $('analyzeBtn').addEventListener('pointerdown', function(e) {
    e.preventDefault();
    $('aiResult').textContent = "Analyzing leaf structure...";
    $('aiResult').style.color = "#ffb020";

    fetch('/analyze')
      .then(response => response.json())
      .then(data => {
        if(data.status === "success") {
          let color = data.disease === "Healthy" ? "#2dff9a" : "#ff3b4e";
          $('aiResult').style.color = color;
          $('aiResult').innerHTML = `Result: ${data.disease} <br> Confidence: ${data.confidence}%`;
        } else {
            $('aiResult').style.color = "#ff3b4e";
            $('aiResult').textContent = data.message;
        }
      })
      .catch(err => {
        $('aiResult').style.color = "#ff3b4e";
        $('aiResult').textContent = "Error processing inference";
      });
  });

  var keys={ArrowUp:'MOVE_FORWARD',w:'MOVE_FORWARD',ArrowDown:'MOVE_BACKWARD',s:'MOVE_BACKWARD',
            ArrowLeft:'TURN_LEFT',a:'TURN_LEFT',ArrowRight:'TURN_RIGHT',d:'TURN_RIGHT'};
  window.addEventListener('keydown',function(e){
    if(e.repeat) return;
    var k=e.key.length===1?e.key.toLowerCase():e.key;
    if(keys[k]){ e.preventDefault(); var el=document.querySelector('.db[data-c="'+keys[k]+'"]'); press(keys[k],el,'kb'); }
    else if(k===' '){ e.preventDefault(); release(); send('STOP'); }
  });
  window.addEventListener('keyup',function(e){
    var k=e.key.length===1?e.key.toLowerCase():e.key;
    if(keys[k]) release('kb');
  });

  document.addEventListener('visibilitychange',function(){ if(document.hidden) release(); });
  window.addEventListener('blur',function(){ release(); });
  setInterval(function(){ if(held===null) send('PING'); },2000);
  document.addEventListener('touchmove',function(e){
    if(e.target.closest('.dpad') || e.target.closest('.stop')) {
        e.preventDefault();
    }
  },{passive:false});
  document.addEventListener('gesturestart',function(e){e.preventDefault()});

  connect();
})();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    # Start the Flask web server in a background thread
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    print("Starting Vision Server and Physical Screen...")
    # Run the physical camera screen loop on the main thread (required by OpenCV)
    camera_loop()

CONFIDENCE_THRESHOLD = 60
def generate_gradcam(model, input_tensor, target_class):

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_layer = model.features[-1]

    handle_f = final_layer.register_forward_hook(forward_hook)
    handle_b = final_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    loss = output[0, target_class]
    loss.backward()

    grads = gradients[-1].detach().cpu().numpy()[0]
    acts = activations[-1].detach().cpu().numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam



def detect_and_classify(image):

    if image is None:
        return None, None, "No image uploaded"

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = yolo_model(img_bgr, conf=0.25, classes=[19])
    result = results[0]

    if len(result.boxes) == 0:
        return image, None, "No cow detected"

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    largest_idx = np.argmax(areas)

    x1,y1,x2,y2 = map(int, boxes[largest_idx])
    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return image, None, "Invalid crop"

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    input_tensor = inference_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():

        # Original
        out1 = model(input_tensor)

        # Horizontal flip
        flipped = torch.flip(input_tensor, dims=[3])
        out2 = model(flipped)

        outputs = (out1 + out2) / 2
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    breed = class_names[pred.item()]
    confidence = confidence.item() * 100

    # Generate GradCAM
    cam = generate_gradcam(model, input_tensor, pred.item())
    cam = cv2.resize(cam, (crop_rgb.shape[1], crop_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    gradcam_img = cv2.addWeighted(crop_rgb, 0.6, heatmap, 0.4, 0)

    annotated = result.plot()

    if confidence < CONFIDENCE_THRESHOLD:
        return annotated[:,:,::-1], gradcam_img, "Low confidence prediction"

    return annotated[:,:,::-1], gradcam_img, f"{breed} ({confidence:.2f}%)"



// DOM元素
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const mainResult = document.getElementById('mainResult');
const healthStatus = document.getElementById('healthStatus');
const predictionBadge = document.getElementById('predictionBadge');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBars = document.getElementById('confidenceBars');
const jsonOutput = document.getElementById('jsonOutput');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const recommendationBox = document.getElementById('recommendationBox');
const recommendationText = document.getElementById('recommendationText');
const diseaseInfo = document.getElementById('diseaseInfo');
const diseaseInfoContent = document.getElementById('diseaseInfoContent');

// 上传区域拖放功能
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

// 点击上传区域
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// 文件选择处理
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelect(e.target.files[0]);
    }
});

// 处理文件选择
function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        showError('请选择叶片图片文件');
        return;
    }

    // 显示预览
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        predictBtn.disabled = false;
        hideError();
        hideMainResult();
    };
    reader.readAsDataURL(file);
}

// 预测按钮点击
predictBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // 显示加载中
    loadingIndicator.style.display = 'block';
    predictBtn.disabled = true;
    hideError();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            displayResult(result);
        } else {
            showError(result.error || '病害诊断失败');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
    } finally {
        loadingIndicator.style.display = 'none';
        predictBtn.disabled = false;
    }
});

// 显示结果
function displayResult(result) {
    // 显示主要结果区域
    mainResult.style.display = 'block';

    const prediction = result.prediction;

    // 更新健康状态
    if (prediction.is_healthy) {
        healthStatus.innerHTML = '<div class="health-status healthy"><i class="fas fa-check-circle me-2"></i>叶片健康</div>';
        predictionBadge.className = 'badge prediction-badge bg-success';
    } else {
        healthStatus.innerHTML = '<div class="health-status disease"><i class="fas fa-exclamation-triangle me-2"></i>发现病害</div>';
        predictionBadge.className = 'badge prediction-badge bg-danger';
    }

    // 更新预测结果徽章
    predictionBadge.textContent = prediction.predicted_class;
    confidenceValue.textContent = (prediction.confidence * 100).toFixed(2) + '%';

    // 显示防治建议
    if (prediction.recommendation) {
        recommendationText.textContent = prediction.recommendation;
        recommendationBox.style.display = 'block';
    } else {
        recommendationBox.style.display = 'none';
    }

    // 显示病害信息
    if (prediction.disease_info && Object.keys(prediction.disease_info).length > 0) {
        let infoHtml = '';
        for (const [key, value] of Object.entries(prediction.disease_info)) {
            infoHtml += `<p><strong>${key}:</strong> ${value}</p>`;
        }
        diseaseInfoContent.innerHTML = infoHtml;
        diseaseInfo.style.display = 'block';
    } else {
        diseaseInfo.style.display = 'none';
    }

    // 生成置信度条
    confidenceBars.innerHTML = '';
    const probabilities = prediction.all_probabilities;

    Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([className, confidence]) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'mb-2';

            const label = document.createElement('div');
            label.className = 'd-flex justify-content-between mb-1';
            label.innerHTML = `
                <span>${className}</span>
                <span>${(confidence * 100).toFixed(1)}%</span>
            `;

            const bar = document.createElement('div');
            bar.className = 'confidence-bar';

            const fill = document.createElement('div');
            fill.className = className.includes('健康') || className.toLowerCase().includes('healthy') ?
                'confidence-fill' : 'confidence-fill disease';
            fill.style.width = '0%';

            bar.appendChild(fill);
            barContainer.appendChild(label);
            barContainer.appendChild(bar);
            confidenceBars.appendChild(barContainer);

            // 延迟设置宽度以触发CSS动画
            setTimeout(() => {
                fill.style.width = (confidence * 100) + '%';
                fill.textContent = (confidence * 100).toFixed(1) + '%';
            }, 100);
        });

    // 美化JSON显示
    jsonOutput.innerHTML = syntaxHighlight(JSON.stringify(result, null, 2));
}

// JSON语法高亮
function syntaxHighlight(json) {
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    function (match) {
        let cls = 'text-primary';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'text-warning';
            } else {
                cls = 'text-success';
            }
        } else if (/true|false/.test(match)) {
            cls = 'text-info';
        } else if (/null/.test(match)) {
            cls = 'text-danger';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

// 显示错误
function showError(message) {
    errorMessage.textContent = message;
    errorAlert.style.display = 'block';
    hideMainResult();
}

// 隐藏错误
function hideError() {
    errorAlert.style.display = 'none';
}

// 隐藏主要结果
function hideMainResult() {
    mainResult.style.display = 'none';
}

// 页面加载完成后检查服务状态
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const result = await response.json();
        if (result.status !== 'healthy') {
            showError('服务状态异常');
        }
    } catch (error) {
        showError('无法连接到病害诊断服务');
    }
});

// 获取应用信息
async function loadAppInfo() {
    try {
        const response = await fetch('/api/info');
        const result = await response.json();
        console.log('应用信息:', result);
    } catch (error) {
        console.error('获取应用信息失败:', error);
    }
}

// 页面加载时获取应用信息
loadAppInfo();
document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const detailsSheet = document.getElementById("detailsSheet");
    const sheetNodeName = document.getElementById("sheetNodeName");
    const statusText = document.getElementById("statusText");
    const offlineNote = document.getElementById("offlineNote");
    const confidenceText = document.getElementById("confidenceText");
    const timestampText = document.getElementById("timestampText");
    const distanceText = document.getElementById("distanceText");
    const spectrogramImage = document.getElementById("spectrogramImage");
    const imageLoader = document.getElementById("imageLoader");
    const statusRow = document.querySelector(".status-row");
    const globalStatus = document.getElementById("globalStatus");
    const infoBtn = document.getElementById("infoBtn");
    const infoWindow = document.getElementById("infoWindow");
    const closeInfoBtn = document.getElementById("closeInfoBtn");
    const userAlertModal = document.getElementById("userAlertModal");
    const closeUserAlert = document.getElementById("closeUserAlert");

    // Info Window Logic
    infoBtn.addEventListener("click", () => {
        infoWindow.classList.add("active");
    });
    closeInfoBtn.addEventListener("click", () => {
        infoWindow.classList.remove("active");
    });

    // Alert Modal Logic
    closeUserAlert.addEventListener("click", () => {
        userAlertModal.style.display = "none";
    });

    let orcaCircles = {};
    let alertedNodes = new Set();
    const vesselsSection = document.getElementById("vesselsSection");
    const vesselsList = document.getElementById("vesselsList");
    const audioPlayBtn = document.getElementById("audioPlayBtn");
    const liveAudio = document.getElementById("liveAudio");

    // State
    let map;
    let hydrophoneMarkers = {};
    let pollingIntervals = {};
    let activeNodeId = null;
    let userLocation = null;
    let hls = null;
    let currentAudioUrl = null;

    // Initialize Map
    function initMap() {
        // Center on Salish Sea and prevent map looping
        const worldBounds = [[-90, -180], [90, 180]];
        map = L.map('map', {
            zoomControl: false,
            maxBounds: worldBounds,
            maxBoundsViscosity: 0.8,
            minZoom: 3
        }).setView([48.2, -122.8], 9);

        // Esri street map tiles
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19,
            minZoom: 3,
            attribution: 'Tiles &copy; Esri',
            noWrap: true
        }).addTo(map);

        // Request user location
        if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(position => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                userLocation = L.latLng(lat, lng);

                // Add pulsing green dot for user
                const userIcon = L.divIcon({
                    className: 'custom-div-icon',
                    html: '<div class="user-marker"></div>',
                    iconSize: [18, 18],
                    iconAnchor: [9, 9]
                });

                L.marker([lat, lng], { icon: userIcon }).addTo(map)
                    .bindPopup("You are here")
                    .openPopup();

                // Slightly adjust view to include user
                map.setView([lat, lng], 10);
            }, err => {
                console.log("Geolocation denied or unavailable:", err);
            });
        }
    }

    // Fetch and Plot Hydrophones
    async function initHydrophones() {
        try {
            const res = await fetch("/api/nodes");
            const data = await res.json();

            data.nodes.forEach(node => {
                // Create custom marker
                const icon = L.divIcon({
                    className: 'custom-div-icon',
                    html: `<div class="hydrophone-marker" id="marker-${node.id}"></div>`,
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                });

                const marker = L.marker([node.lat, node.lng], { icon: icon }).addTo(map);

                // Marker Click Event
                marker.on('click', () => {
                    openDetailsSheet(node);
                });

                hydrophoneMarkers[node.id] = {
                    node: node,
                    marker: marker,
                    lastStatus: null,
                    consecutiveDetections: 0
                };

                // Start polling this node
                pollNodeStatus(node.id);
                pollingIntervals[node.id] = setInterval(() => pollNodeStatus(node.id), 10000);
            });
        } catch (err) {
            console.error("Failed to load hydrophones", err);
            globalStatus.textContent = "API Error";
            globalStatus.style.color = "var(--accent-red)";
        }
    }

    // Background polling for a specific node
    async function pollNodeStatus(nodeId) {
        try {
            const res = await fetch(`/api/status?node=${nodeId}`);
            const data = await res.json();

            const markerDiv = document.getElementById(`marker-${nodeId}`);

            // Debounce transient noise: require 2 consecutive positive detections
            if (!data.error && data.orca_detected) {
                hydrophoneMarkers[nodeId].consecutiveDetections += 1;
            } else {
                hydrophoneMarkers[nodeId].consecutiveDetections = 0;
            }

            const isConfirmedDetection = hydrophoneMarkers[nodeId].consecutiveDetections >= 2;
            data.orca_detected = isConfirmedDetection;

            const node = hydrophoneMarkers[nodeId].node;

            if (isConfirmedDetection) {
                // Trigger Map Animation
                if (markerDiv) markerDiv.classList.add("alert");

                // Draw circle if not exists
                if (!orcaCircles[nodeId]) {
                    orcaCircles[nodeId] = L.circle([node.lat, node.lng], {
                        color: '#ef4444',
                        fillColor: '#ef4444',
                        fillOpacity: 0.15,
                        weight: 2,
                        radius: 6500, // 6.5 km
                        className: 'flaring-circle',
                        interactive: false
                    }).addTo(map);
                }

                // Check distance to user
                if (userLocation && !alertedNodes.has(nodeId)) {
                    const nodeLoc = L.latLng(node.lat, node.lng);
                    const dist = map.distance(userLocation, nodeLoc);
                    if (dist <= 6500) {
                        userAlertModal.style.display = "flex";
                        alertedNodes.add(nodeId);
                    }
                }
            } else {
                if (markerDiv) markerDiv.classList.remove("alert");

                // Remove circle if exists
                if (orcaCircles[nodeId]) {
                    map.removeLayer(orcaCircles[nodeId]);
                    delete orcaCircles[nodeId];
                }
                alertedNodes.delete(nodeId);
            }

            // Handle offline / online map marker appearance
            if (data.error || data.offline) {
                if (markerDiv) markerDiv.classList.add("offline");
            } else {
                if (markerDiv) markerDiv.classList.remove("offline");
            }

            // Save status for UI updating
            hydrophoneMarkers[nodeId].lastStatus = data;

            // Re-evaluate global status pill & corner alert across all nodes
            // If multiple hydrophones detect an orca, select the one closest to the user
            let anyOrcaDetected = false;
            let closestDetectingNode = null;
            let shortestDistance = Infinity;

            for (const key in hydrophoneMarkers) {
                const ns = hydrophoneMarkers[key].lastStatus;
                if (ns && !ns.error && !ns.offline && ns.orca_detected) {
                    anyOrcaDetected = true;
                    const hNode = hydrophoneMarkers[key].node;

                    if (userLocation) {
                        const hDist = map.distance(userLocation, L.latLng(hNode.lat, hNode.lng));
                        if (hDist < shortestDistance) {
                            shortestDistance = hDist;
                            closestDetectingNode = hNode;
                        }
                    } else if (!closestDetectingNode) {
                        // Fallback if user geolocation is unavailable or denied
                        closestDetectingNode = hNode;
                    }
                }
            }

            if (anyOrcaDetected) {
                globalStatus.textContent = "ORCA DETECTED";
                globalStatus.classList.add("alert");
                globalStatus.style.cursor = "pointer";
                globalStatus.title = "Click to view closest detecting hydrophone";
                globalStatus.onclick = () => {
                    if (closestDetectingNode) openDetailsSheet(closestDetectingNode);
                };
            } else {
                globalStatus.textContent = "No Orca Detected";
                globalStatus.classList.remove("alert");
                globalStatus.style.cursor = "default";
                globalStatus.title = "";
                globalStatus.onclick = null;
            }

            // If this node's sheet is currently open, update the sheet UI live
            if (activeNodeId === nodeId) {
                updateSheetUI(data);
            }
        } catch (err) {
            console.error(`Error polling ${nodeId}`, err);
        }
    }

    // Open Bottom Sheet
    function openDetailsSheet(node) {
        if (activeNodeId) {
            const oldMarker = document.getElementById(`marker-${activeNodeId}`);
            if (oldMarker) oldMarker.classList.remove("active");
        }

        activeNodeId = node.id;
        sheetNodeName.textContent = node.name;

        const newMarker = document.getElementById(`marker-${node.id}`);
        if (newMarker) newMarker.classList.add("active");

        // Reset UI temporarily
        statusText.textContent = "Loading feed...";
        if (offlineNote) offlineNote.style.display = "none";
        confidenceText.textContent = "Confidence: --";
        timestampText.textContent = "Sync: --";
        distanceText.textContent = "Distance: --";
        spectrogramImage.style.display = "none";
        imageLoader.textContent = "Loading stream...";
        imageLoader.style.animation = "blink 1.5s infinite";
        imageLoader.style.display = "block";
        statusRow.classList.remove("danger");
        statusRow.classList.remove("offline");
        vesselsSection.style.display = "none";
        vesselsList.innerHTML = "";

        // Pan map up slightly so marker isn't covered by sheet
        map.setView([node.lat - 0.1, node.lng], map.getZoom(), { animate: true });

        // Slide sheet up
        detailsSheet.classList.add("active");

        // Calculate Distance
        if (userLocation) {
            const nodeLocation = L.latLng(node.lat, node.lng);
            const distMeters = map.distance(userLocation, nodeLocation);
            distanceText.textContent = `Distance: ${(distMeters / 1000).toFixed(1)} km`;
        } else {
            distanceText.textContent = `Distance: Unknown`;
        }

        // Populate immediately with cached status if available
        if (hydrophoneMarkers[node.id].lastStatus) {
            updateSheetUI(hydrophoneMarkers[node.id].lastStatus);
        }
    }

    // Update Bottom Sheet DOM
    function updateSheetUI(data) {
        if (data.error || data.offline) {
            statusText.textContent = "Offline";
            statusRow.classList.remove("danger");
            statusRow.classList.add("offline");
            if (offlineNote) offlineNote.style.display = "block";
            confidenceText.textContent = "Confidence: N/A";
            timestampText.textContent = "Sync: Offline";
            
            imageLoader.style.display = "block";
            imageLoader.textContent = "No live acoustic feed available (node offline)";
            imageLoader.style.animation = "none";
            spectrogramImage.style.display = "none";
            
            audioPlayBtn.style.display = "none";
            if (audioPlayBtn.classList.contains("playing")) {
                stopAudioStream();
            }
            
            vesselsSection.style.display = "none";
            vesselsList.innerHTML = "";
            return;
        }

        // Online node handling
        statusRow.classList.remove("offline");
        if (offlineNote) offlineNote.style.display = "none";
        imageLoader.style.animation = "blink 1.5s infinite";
        imageLoader.textContent = "Loading stream...";

        if (data.orca_detected) {
            statusRow.classList.add("danger");
            statusText.textContent = "ORCA DETECTED";
        } else {
            statusRow.classList.remove("danger");
            statusText.textContent = "Listening - Clear";
        }

        confidenceText.textContent = `Confidence: ${(data.detection_confidence * 100).toFixed(1)}%`;

        if (data.timestamp) {
            const date = new Date(data.timestamp);
            timestampText.textContent = `Sync: ${date.toLocaleTimeString()}`;
        } else {
            timestampText.textContent = `Sync: --`;
        }

        if (data.spectrogram_b64) {
            imageLoader.style.display = "none";
            spectrogramImage.style.display = "block";
            spectrogramImage.src = `data:image/png;base64,${data.spectrogram_b64}`;
        }
        
        if (data.audio_url) {
            audioPlayBtn.style.display = "flex";
            audioPlayBtn.dataset.url = data.audio_url;
            
            // Switch streams if the station changed while playing
            if (audioPlayBtn.classList.contains("playing") && currentAudioUrl !== data.audio_url) {
                startAudioStream(data.audio_url);
            }
        } else {
            audioPlayBtn.style.display = "none";
            if (audioPlayBtn.classList.contains("playing")) {
                stopAudioStream();
            }
        }

        // Update vessel telemetry if Orca is detected (and vessels are tracked)
        if (data.orca_detected) {
            vesselsSection.style.display = "block";
            vesselsList.innerHTML = "";

            if (data.vessels && data.vessels.length > 0) {
                data.vessels.forEach(v => {
                    const card = document.createElement("div");
                    card.className = "vessel-card";

                    const header = document.createElement("div");
                    header.className = "vessel-header";

                    const nameSpan = document.createElement("span");
                    nameSpan.className = "vessel-name";
                    nameSpan.textContent = v.name || "Unknown Vessel";

                    const speedSpan = document.createElement("span");
                    speedSpan.className = "vessel-speed";
                    speedSpan.textContent = `${v.sog ? v.sog.toFixed(1) : "0.0"} kts`;

                    header.appendChild(nameSpan);
                    header.appendChild(speedSpan);
                    card.appendChild(header);

                    const grid = document.createElement("div");
                    grid.className = "vessel-grid";

                    const metrics = [
                        { label: "Status", value: v.status || "Unknown" },
                        { label: "Distance", value: `${v.distance_km ? v.distance_km.toFixed(2) : "0.00"} km` },
                        { label: "Acoustic Noise", value: `${v.source_level_db ? v.source_level_db.toFixed(1) : "0.0"} dB` },
                        { label: "Impact Area", value: `${v.reclaimed_area_km2 ? v.reclaimed_area_km2.toFixed(2) : "0.00"} km²` }
                    ];

                    metrics.forEach(m => {
                        const metricDiv = document.createElement("div");
                        metricDiv.className = "vessel-metric";

                        const labelSpan = document.createElement("span");
                        labelSpan.className = "vessel-metric-label";
                        labelSpan.textContent = m.label;

                        const valSpan = document.createElement("span");
                        valSpan.className = "vessel-metric-value";
                        valSpan.textContent = m.value;

                        metricDiv.appendChild(labelSpan);
                        metricDiv.appendChild(valSpan);
                        grid.appendChild(metricDiv);
                    });

                    card.appendChild(grid);
                    vesselsList.appendChild(card);
                });
            } else {
                const emptyMsg = document.createElement("div");
                emptyMsg.className = "loader";
                emptyMsg.style.cssText = "text-align: center; padding: 1rem; font-size: 0.8rem; animation: none;";
                emptyMsg.textContent = "No commercial vessels detected in hydrophone range.";
                vesselsList.appendChild(emptyMsg);
            }
        } else {
            vesselsSection.style.display = "none";
            vesselsList.innerHTML = "";
        }
    }

    // Run initialization
    initMap();
    initHydrophones();

    function startAudioStream(url) {
        if (!url) return;
        currentAudioUrl = url;
        if (Hls.isSupported()) {
            if (hls) hls.destroy();
            hls = new Hls();
            hls.loadSource(url);
            hls.attachMedia(liveAudio);
            hls.on(Hls.Events.MANIFEST_PARSED, function () {
                liveAudio.play();
            });
        } else if (liveAudio.canPlayType('application/vnd.apple.mpegurl')) {
            liveAudio.src = url;
            liveAudio.play();
        }
        audioPlayBtn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
        audioPlayBtn.classList.add("playing");
    }

    function stopAudioStream() {
        if (hls) {
            hls.destroy();
            hls = null;
        }
        liveAudio.pause();
        currentAudioUrl = null;
        audioPlayBtn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
        audioPlayBtn.classList.remove("playing");
    }

    function closeDetailsSheet() {
        detailsSheet.classList.remove("active");
        if (activeNodeId) {
            const markerDiv = document.getElementById(`marker-${activeNodeId}`);
            if (markerDiv) markerDiv.classList.remove("active");
        }
        activeNodeId = null;
        stopAudioStream();
    }

    // Audio Player Logic
    audioPlayBtn.addEventListener("click", () => {
        const url = audioPlayBtn.dataset.url;
        if (!url) return;

        if (audioPlayBtn.classList.contains("playing")) {
            stopAudioStream();
        } else {
            startAudioStream(url);
        }
    });

    // Close sheet when tapping map background
    map.on('click', closeDetailsSheet);

    // Also close on handle swipe/click
    const sheetHandle = document.querySelector('.sheet-handle');
    sheetHandle.addEventListener('click', closeDetailsSheet);

    // Touch swipe down to close
    let touchStartY = 0;
    sheetHandle.addEventListener('touchstart', (e) => {
        touchStartY = e.touches[0].clientY;
    });
    sheetHandle.addEventListener('touchend', (e) => {
        const touchEndY = e.changedTouches[0].clientY;
        if (touchEndY > touchStartY + 30) { // 30px swipe down threshold
            closeDetailsSheet();
        }
    });
});

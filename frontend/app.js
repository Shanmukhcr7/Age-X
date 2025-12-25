import { db, auth, googleProvider, signInWithPopup, signOut, onAuthStateChanged } from "./firebase.js";
import { collection, getDocs, query, where, limit, startAfter, orderBy } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    API_URL: "http://localhost:8000/api/age-check",
    SAFETY: {
        CONFIDENCE_THRESHOLD: 0.60,
        ADULT_FRAME_BUFFER: 1, // Unlock faster (20s interval is long enough)
        KID_LOCK_IMMEDIATE: true
    },
    INTERVAL_MS: 20000 // 20 Seconds
};

// ============================================
// ROUTER (VIEW MANAGER)
// ============================================
class Router {
    constructor(app) {
        this.app = app;
        this.views = document.querySelectorAll(".view");
        this.navItems = document.querySelectorAll(".nav-item");

        this.currentView = "login"; // Default
        this.scrollStore = {};

        // Nav Click
        this.navItems.forEach(item => {
            item.addEventListener("click", () => {
                const target = item.dataset.target;
                if (target) this.navigate(target);
            });
        });

        // Hide Nav on Login
        this.navBar = document.querySelector(".bottom-nav");
    }

    navigate(route) {
        if (route === this.currentView) return;

        // Guard: Auth
        if (!auth.currentUser && route !== "login") {
            console.warn("Auth Locked. Redirecting to Login.");
            route = "login";
        }

        // Save Scroll of current view
        const currentEl = document.getElementById(`view-${this.currentView}`);
        if (currentEl) this.scrollStore[this.currentView] = currentEl.scrollTop;

        // Update State
        this.currentView = route;

        // 1. Update Views
        this.views.forEach(view => {
            if (view.id === `view-${route}`) {
                view.classList.add("active-view");
                gsap.fromTo(view, { opacity: 0, x: 20 }, { opacity: 1, x: 0, duration: 0.4 });
                // Trigger On-Load events for views
                if (route === "reels") this.app.reelManager.playCurrent();
                if (route === "home") this.app.homeManager.loadSections(this.app.safetyEngine.currentMode, true); // Pass current mode and reset
                if (route === "profile") this.app.profileManager.updateUI();

                // Restore Scroll
                if (this.scrollStore[route] !== undefined) {
                    setTimeout(() => view.scrollTop = this.scrollStore[route], 50); // Small delay for rendering
                }
            } else {
                view.classList.remove("active-view");
                // Pause Reels if leaving
                if (view.id === "view-reels") this.app.reelManager.pause();
            }
        });

        // 2. Update Nav State
        this.navItems.forEach(item => {
            if (item.dataset.target === route) item.classList.add("active-nav");
            else item.classList.remove("active-nav");
        });

        // 3. Show/Hide Nav Bar
        if (route === "login") {
            gsap.to(this.navBar, { y: 100, opacity: 0, duration: 0.3 });
        } else {
            gsap.to(this.navBar, { y: 0, opacity: 1, duration: 0.3 });
        }
    }
}

// ============================================
// MANAGERS
// ============================================

class ProfileManager {
    constructor() {
        this.nameEl = document.getElementById("profileName");
        this.emailEl = document.getElementById("profileEmail");
        this.avatarEl = document.getElementById("profileAvatar");
        this.btnLogout = document.getElementById("btnLogout");

        this.btnLogout.addEventListener("click", () => {
            signOut(auth).then(() => window.location.reload());
        });
    }

    updateUI() {
        const user = auth.currentUser;
        if (user) {
            this.nameEl.innerText = user.displayName;
            this.emailEl.innerText = user.email;
            if (user.photoURL) {
                this.avatarEl.innerHTML = `<img src="${user.photoURL}" alt="Profile">`;
            }
        }
    }
}





class HomeManager {
    constructor(app) {
        this.app = app;
        this.grid = document.getElementById("homeGrid");
        this.loading = false;
        // Remove class content-grid to allow custom layout
        this.grid.className = "home-content";
        this.currentMode = "Kid"; // Default
    }

    async loadSections(mode = null, reset = false) {
        if (mode) this.currentMode = mode;
        if (this.loading) return;
        this.loading = true;
        if (reset) {
            this.grid.innerHTML = "";
        }

        try {
            // Define categories based on mode
            const categories = this.currentMode === "Adult"
                ? ["Kid", "Teen", "Adult", "Senior", "Young Adult", "adult", "teen"]
                : ["Kid", "kid"];

            // Get Videos (Limit 20 for sections)
            const q = query(
                collection(db, "reels"),
                where("category", "in", categories),
                limit(20)
            );
            const snapshot = await getDocs(q);
            const videos = [];
            snapshot.forEach(doc => videos.push(doc.data()));

            console.log(`HOME: Found ${videos.length} videos for ${this.currentMode}`);

            // Split into sections
            if (videos.length > 0) {
                const trending = videos.slice(0, 10);
                const liked = videos.slice(10, 20).length ? videos.slice(10, 20) : videos.slice(0, 5); // Fallback

                this.renderSection("🔥 Trending Now", trending, "trending");
                this.renderSection("❤️ Most Liked", liked, "liked");
            } else {
                this.renderFallback();
            }

        } catch (e) {
            console.error("Home Load Error", e);
            if (reset) this.renderFallback();
        }
        this.loading = false;
    }

    renderSection(title, videos, type) {
        // Section Container
        const section = document.createElement("div");
        section.className = "home-section";

        // Title
        const heading = document.createElement("h3");
        heading.className = "section-title";
        heading.innerHTML = type === 'trending' ? `<i data-lucide="flame"></i> ${title}` : `<i data-lucide="heart"></i> ${title}`;
        section.appendChild(heading);

        // Scroll Container
        const scrollRow = document.createElement("div");
        scrollRow.className = "horizontal-scroll";

        videos.forEach(data => {
            this.renderCard(data, scrollRow);
        });

        section.appendChild(scrollRow);
        this.grid.appendChild(section);
        if (window.lucide) lucide.createIcons();
    }

    renderCard(data, container) {
        const videoId = this.extractVideoId(data.url);
        if (!videoId) return;

        // Mock Stats
        const views = Math.floor(Math.random() * 900) + 100 + "K";
        const likes = (Math.random() * 9).toFixed(1) + "K";

        const card = document.createElement("div");
        card.className = "card-item glass-panel";
        card.innerHTML = `
            <img src="https://img.youtube.com/vi/${videoId}/hqdefault.jpg" loading="lazy">
            <div class="card-tag">HD</div>
            <div class="card-overlay">
                <div style="font-weight:600; font-size: 0.9rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    ${data.title || "Watch Video"}
                </div>
                <div class="card-stats">
                    <span><i data-lucide="eye" style="width:12px"></i> ${views}</span>
                    <span><i data-lucide="thumbs-up" style="width:12px"></i> ${likes}</span>
                </div>
            </div>
        `;

        card.addEventListener("click", () => {
            if (this.app.reelManager) this.app.reelManager.playVideoId(videoId);
            this.app.router.navigate("reels");
        });

        container.appendChild(card);
    }

    extractVideoId(url) {
        let id = "";
        if (url.includes("shorts/")) id = url.split("shorts/")[1].split("?")[0];
        else if (url.includes("v=")) id = url.split("v=")[1].split("&")[0];
        else if (url.includes("youtu.be/")) id = url.split("youtu.be/")[1];
        return id;
    }

    renderFallback() {
        // Fallback or Demo content
        const mocks = [
            { url: "https://youtube.com/watch?v=5qap5aO4i9A", title: "Lo-Fi Beats" },
            { url: "https://youtube.com/watch?v=tgbNymZ7vqY", title: "Coding Life" },
            { url: "https://youtube.com/watch?v=jfKfPfyJRdk", title: "Relaxing View" }
        ];
        // Just render them into a specific section if empty
        if (this.grid.children.length === 0) {
            this.renderSection("Recommended", mocks, "trending");
        }
    }

    // Alias for compatibility
    loadGrid(mode, reset) { this.loadSections(mode, reset); }
}

// ... SAFETY & REEL LOGIC Same as before, but wrapped ...

class ReelManager {
    constructor(app) {
        this.app = app;
        this.reels = [];
        this.currentIndex = 0;
        this.player = document.getElementById("reelPlayer");
        this.container = document.getElementById("reelContainer");
    }

    playVideoId(id) {
        const url = this.formatUrl(`https://youtube.com/watch?v=${id}`);
        if (url) {
            // Add to front of queue or just play? 
            // Better: Play immediately, but keep queue for scrolling.
            // Let's insert it at current index or replace current?
            // User wants "scroll to next video". So we should insert at current index.
            this.reels.splice(this.currentIndex, 0, url); // Insert at current
            this.playCurrent(); // Play it
        }
    }

    pause() {
        // Pause via YouTube API
        if (this.player.contentWindow) {
            this.player.contentWindow.postMessage('{"event":"command","func":"pauseVideo","args":""}', '*');
        }
    }
    play() {
        if (this.player.contentWindow) {
            this.player.contentWindow.postMessage('{"event":"command","func":"playVideo","args":""}', '*');
        }
    }

    // (Existing loadReels, playCurrent logic from previous step, slightly adapted)
    async loadReels(mode) {
        console.log(`REELS: Loading for ${mode}...`);
        const categories = mode === "Adult"
            ? ["Kid", "Teen", "Adult", "Senior", "Young Adult", "adult", "teen", "senior"]
            : ["Kid", "kid"];

        try {
            const q = query(
                collection(db, "reels"),
                where("category", "in", categories)
            );
            const snapshot = await getDocs(q);
            const newReels = [];
            snapshot.forEach(doc => {
                const data = doc.data();
                if (data.url) newReels.push(this.formatUrl(data.url));
            });

            console.log(`REELS: Found ${newReels.length} videos for ${mode}`);

            if (newReels.length === 0) this.reels = this.getFallbackReels(mode);
            else this.reels = this.shuffle(newReels);

            this.currentIndex = 0;
            // Only play if actively looking at reels
            if (document.querySelector(".view.active-view")?.id === "view-reels") {
                this.playCurrent();
            }
        } catch (e) {
            console.warn("REELS: loading fallback error", e);
            this.reels = this.getFallbackReels(mode);
        }
    }

    formatUrl(url) {
        let id = "";
        if (url.includes("shorts/")) id = url.split("shorts/")[1].split("?")[0];
        else if (url.includes("v=")) id = url.split("v=")[1].split("&")[0];
        else if (url.includes("youtu.be/")) id = url.split("youtu.be/")[1];
        // Loop single video: loop=1&playlist={VIDEO_ID}
        return id ? `https://www.youtube-nocookie.com/embed/${id}?autoplay=1&controls=0&playsinline=1&enablejsapi=1&loop=1&playlist=${id}` : null;
    }

    playCurrent() {
        if (this.reels.length === 0) return;
        this.player.src = this.reels[this.currentIndex];
    }

    next() {
        this.currentIndex = (this.currentIndex + 1) % this.reels.length;
        this.playCurrent();
    }
    prev() {
        this.currentIndex = (this.currentIndex - 1 + this.reels.length) % this.reels.length;
        this.playCurrent();
    }
    shuffle(array) { return array.sort(() => Math.random() - 0.5); }
    getFallbackReels(mode) {
        const safe = ["https://www.youtube-nocookie.com/embed/5qap5aO4i9A?autoplay=1&controls=0"];
        const adult = [...safe, "https://www.youtube-nocookie.com/embed/jfKfPfyJRdk?autoplay=1&controls=0"];
        return mode === "Adult" ? adult : safe;
    }
}

class SafetyEngine {
    constructor(onModeChange) {
        this.currentMode = "Kid";
        this.consecutiveAdultFrames = 0;
        this.errorBuffer = 0; // Grace period for missing faces
        this.MAX_ERROR_BUFFER = 3; // Allow 3 frames (~4.5s) of "No Face" before locking
        this.onModeChange = onModeChange;
    }

    processResult(data) {
        console.log("Detection:", data.age_group, data.confidence);
        // 1. Handle Errors (No Face, Network Error, etc)
        if (data.error || data.forced_safety) {
            console.warn("Safety Warning:", data.msg || "Unknown Error");

            if (this.currentMode === "Adult") {
                this.errorBuffer++;
                if (this.errorBuffer > this.MAX_ERROR_BUFFER) {
                    this.forceSafeMode("Face Lost / Error Limit Reached");
                }
            }
            return;
        }

        // Reset Error Buffer on any valid detection
        this.errorBuffer = 0;

        const detectedGroup = data.age_group;
        const isSafeGroup = ["Kid", "Teen"].includes(detectedGroup);

        if (isSafeGroup) {
            // CRITICAL: Immediate Downgrade on Child Detection
            if (this.currentMode !== "Kid") {
                this.forceSafeMode(`Child Detected (${detectedGroup})`);
            }
            this.consecutiveAdultFrames = 0;
        } else {
            // Adult Detected
            if (this.currentMode === "Kid") {
                if (data.confidence >= CONFIG.SAFETY.CONFIDENCE_THRESHOLD) {
                    this.consecutiveAdultFrames++;
                    if (this.consecutiveAdultFrames >= CONFIG.SAFETY.ADULT_FRAME_BUFFER) {
                        this.unlockAdultMode();
                    }
                } else {
                    this.consecutiveAdultFrames = 0;
                }
            }
        }
    }

    forceSafeMode(reason) {
        this.currentMode = "Kid";
        this.consecutiveAdultFrames = 0;
        this.errorBuffer = 0;
        this.onModeChange("Kid", reason);
    }

    unlockAdultMode() {
        if (this.currentMode !== "Adult") {
            this.currentMode = "Adult";
            this.onModeChange("Adult", "Verified");
        }
    }
}

// ============================================
// MAIN APP CONTROLLER
// ============================================
class App {
    constructor() {
        this.videoElement = document.getElementById("cameraFeed");
        this.canvas = document.getElementById("processingCanvas");
        this.ctx = this.canvas.getContext("2d");

        this.ui = {
            badge: document.getElementById("statusBadge"),
            label: document.getElementById("ageLabel"),
            splash: document.getElementById("splashScreen"),
            appContainer: document.querySelector(".app-container"),
            overlay: document.getElementById("noFaceOverlay"),
            scanOverlay: document.getElementById("scanningOverlay")
        };

        // Managers
        this.reelManager = new ReelManager(this);
        this.homeManager = new HomeManager(this);
        this.profileManager = new ProfileManager();
        this.router = new Router(this);
        this.safetyEngine = new SafetyEngine(this.handleModeChange.bind(this));

        this.init();
    }

    async init() {
        this.animateIntro();

        // Camera
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            this.videoElement.srcObject = stream;
        } catch (e) { console.error("Camera denied", e); }

        // Start Safety Loop
        this.startAgeDetection();

        // Auth Listener
        onAuthStateChanged(auth, (user) => {
            if (user) {
                console.log("Logged In:", user.email);
                this.router.navigate("reels"); // Go to Reels on login
            } else {
                console.log("Logged Out");
                this.router.navigate("login");
            }
        });

        // Login Button
        document.getElementById("btnLogin").addEventListener("click", () => {
            signInWithPopup(auth, googleProvider).catch(err => alert(err.message));
        });

        // Logout Button
        const btnLogout = document.getElementById("btnLogout");
        if (btnLogout) {
            btnLogout.addEventListener("click", () => {
                signOut(auth).catch((error) => console.error("Sign Out Error", error));
            });
        }

        // Camera Toggle
        const btnToggle = document.getElementById("btnCameraToggle");
        if (btnToggle) {
            btnToggle.addEventListener("click", () => this.toggleCameraVisibility());
        }

        this.setupGestures();

        // Initial State: Scanning
        this.showScanning(true);
    }

    showScanning(show) {
        if (this.ui.scanOverlay) {
            if (show) this.ui.scanOverlay.classList.add("scanning-visible");
            else this.ui.scanOverlay.classList.remove("scanning-visible");
        }
    }

    animateIntro() {
        const tl = gsap.timeline();
        tl.to(".splash-logo", { opacity: 1, y: 0, duration: 1 })
            .to(".loader-line", { width: "100%", duration: 1 })
            .to(this.ui.splash, { opacity: 0, duration: 0.5, onComplete: () => this.ui.splash.style.display = "none" })
            .to(this.ui.appContainer, { opacity: 1, duration: 0.5 });
    }

    startAgeDetection() {
        // Define the detection routine
        const runDetection = async () => {
            if (this.videoElement.readyState === 4) {
                const imgData = this.captureFrame();
                if (imgData) {
                    try {
                        const response = await fetch(CONFIG.API_URL, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ image: imgData })
                        });
                        const data = await response.json();
                        this.safetyEngine.processResult(data);
                    } catch (e) {
                        // Only force safe if it's a real recurring network error, 
                        // mostly handled by safety engine, but here we can log.
                        this.safetyEngine.forceSafeMode("Network Error");
                    }
                }
            }
        };

        // 1. Initial Scan Logic
        const initScan = async () => {
            // Wait for camera ready
            if (this.videoElement.readyState !== 4) {
                setTimeout(initScan, 500);
                return;
            }

            try {
                const imgData = this.captureFrame();
                if (imgData) {
                    const response = await fetch(CONFIG.API_URL, {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image: imgData })
                    });
                    const data = await response.json();

                    // Unlock UI
                    this.showScanning(false);

                    // FAST TRACK: Bypass buffer for initial scan if clear winner
                    const isAdult = !["Kid", "Teen"].includes(data.age_group);
                    const isConfident = data.confidence >= CONFIG.SAFETY.CONFIDENCE_THRESHOLD;

                    if (isAdult && isConfident) {
                        this.safetyEngine.currentMode = "Adult"; // Force state
                        this.handleModeChange("Adult", "Scan Complete (Fast Track)");
                    } else {
                        // Standard Fallback (Likely Kid or Low Confidence)
                        this.safetyEngine.processResult(data);
                        this.handleModeChange(this.safetyEngine.currentMode, "Scan Complete");
                    }

                    // Start Periodic Checks
                    setInterval(runDetection, CONFIG.INTERVAL_MS);
                } else {
                    setTimeout(initScan, 500); // Retry if bad frame
                }
            } catch (e) {
                console.error("Init Scan Error", e);
                setTimeout(initScan, 1000); // Retry on error
            }
        };

        // Start waiting for camera
        setTimeout(initScan, 1500);
    }

    captureFrame() {
        // Use reasonable resolution for Face Detection (Haar/SSD needs pixels)
        // 480px is a good balance between speed and accuracy
        this.canvas.width = 480;
        this.canvas.height = 480;
        this.ctx.drawImage(this.videoElement, 0, 0, 480, 480);
        // Compress to 0.7 quality JPEG to reduce payload
        return this.canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    }

    handleModeChange(mode, reason) {
        console.log(`MODE: ${mode} (${reason})`);
        if (mode === "Kid") {
            this.ui.badge.className = "status-badge status-kid";
            this.ui.label.innerText = "Kid Safe";

            // Check for Face Lost ONLY (Child Detection should just switch mode)
            if (reason && reason.includes("Face Lost")) {
                if (this.ui.overlay) this.ui.overlay.classList.add("no-face-visible");
                this.reelManager.pause();
                return; // Stop here, don't load content
            }
        } else {
            this.ui.badge.className = "status-badge status-adult";
            this.ui.label.innerText = "Adult Mode";
        }

        // Resume if valid
        if (this.ui.overlay) this.ui.overlay.classList.remove("no-face-visible");
        // this.reelManager.play(); // REMOVED: Don't play old video before loading new ones
        this.reelManager.loadReels(mode);
        // Ensure Home Grid also updates with relevant content (Resetting grid)
        this.homeManager.loadGrid(mode, true);
    }

    toggleCameraVisibility() {
        const wrapper = document.querySelector(".camera-wrapper");
        const btn = document.getElementById("btnCameraToggle");
        if (wrapper) {
            wrapper.classList.toggle("camera-hidden");
            const isHidden = wrapper.classList.contains("camera-hidden");
            // Update icon
            if (btn) {
                btn.innerHTML = isHidden ? `<i data-lucide="video-off"></i>` : `<i data-lucide="video"></i>`;
                if (window.lucide) lucide.createIcons();
            }
        }
    }

    setupGestures() {
        // Swipe on Reels
        let startY = 0;
        const container = document.getElementById("reelContainer");
        container.addEventListener("touchstart", e => startY = e.touches[0].clientY);
        container.addEventListener("touchend", e => {
            const diff = startY - e.changedTouches[0].clientY;
            if (Math.abs(diff) > 50) diff > 0 ? this.reelManager.next() : this.reelManager.prev();
        });

        // Icons
        if (window.lucide) window.lucide.createIcons();
    }
}

window.addEventListener("load", () => new App());

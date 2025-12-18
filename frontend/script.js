/*************************************************
 * IMPORTS
 *************************************************/
import { db } from "./firebase.js";
import {
  collection,
  getDocs,
  query,
  where
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

import { gsap } from "https://cdn.skypack.dev/gsap@3.12.2";

/*************************************************
 * DOM ELEMENTS
 *************************************************/
const reelContainer = document.getElementById("reelContainer");
const reelPlayer = document.getElementById("reelPlayer");
const camera = document.getElementById("camera");
const canvas = document.getElementById("captureCanvas");
const ageStatus = document.getElementById("ageStatus");

const nextBtn = document.getElementById("nextBtn");
const prevBtn = document.getElementById("prevBtn");

const ctx = canvas.getContext("2d");

/*************************************************
 * GLOBAL STATE
 *************************************************/
let reels = [];
let currentIndex = 0;
let lockedAgeGroup = null;
let videoLoadTimeout = null;
let userInteracted = false;

/*************************************************
 * AGE PRIORITY (UPGRADE ONLY)
 *************************************************/
const AGE_PRIORITY = {
  "Kid": 1,
  "Teen": 2,
  "Young Adult": 3,
  "Adult": 4,
  "Senior": 5
};

/*************************************************
 * CAMERA START
 *************************************************/
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    camera.srcObject = stream;
    camera.play();

    gsap.from(camera, {
      scale: 0,
      rotation: 180,
      duration: 1,
      ease: "back.out(1.7)"
    });
  })
  .catch(() => {
    ageStatus.innerText = "Camera permission required";
  });

/*************************************************
 * CATEGORY RULES
 *************************************************/
function getAllowedCategories(ageGroup) {
  switch (ageGroup) {
    case "Kid": return ["Kid"];
    case "Teen": return ["Kid", "Teen"];
    case "Young Adult": return ["Kid", "Teen", "Young Adult"];
    case "Adult": return ["Kid", "Teen", "Young Adult", "Adult", "Senior"];
    case "Senior": return ["Senior"];
    default: return ["Kid"];
  }
}

/*************************************************
 * CACHE
 *************************************************/
const CACHE_TTL = 60 * 60 * 1000;

function cacheKey(group) {
  return `reels_cache_${group}`;
}

function saveToCache(group, data) {
  localStorage.setItem(
    cacheKey(group),
    JSON.stringify({ reels: data, time: Date.now() })
  );
}

function loadFromCache(group) {
  const raw = localStorage.getItem(cacheKey(group));
  if (!raw) return null;

  const parsed = JSON.parse(raw);
  if (Date.now() - parsed.time > CACHE_TTL) {
    localStorage.removeItem(cacheKey(group));
    return null;
  }
  return parsed.reels;
}

/*************************************************
 * LOAD REELS
 *************************************************/
async function loadReels(ageGroup) {
  reels = [];
  currentIndex = 0;

  const cached = loadFromCache(ageGroup);
  if (cached?.length) {
    reels = cached;
    playReel(0, true);
    return;
  }

  const q = query(
    collection(db, "reels"),
    where("category", "in", getAllowedCategories(ageGroup))
  );

  const snap = await getDocs(q);
  snap.forEach(doc => {
    if (doc.data().url) reels.push(doc.data().url);
  });

  if (!reels.length) return;

  saveToCache(ageGroup, reels);
  playReel(0, true);
}

/*************************************************
 * PLAY REEL (SAFE)
 *************************************************/
function playReel(index, instant = false) {
  if (!reels[index]) return;

  clearTimeout(videoLoadTimeout);

  const embedUrl = convertToEmbed(reels[index]);
  if (!embedUrl) {
    console.warn("Invalid embed URL, skipping");
    nextReel();
    return;
  }

  if (!instant) {
    gsap.to(reelContainer, {
      opacity: 0,
      y: -30,
      duration: 0.35,
      ease: "power2.inOut",
      onComplete: () => {
        reelPlayer.src = embedUrl;
        animateIn();
      }
    });
  } else {
    reelPlayer.src = embedUrl;
    animateIn();
  }

  // 🔐 BLOCKED VIDEO DETECTION (NOT AUTO ADVANCE)
  videoLoadTimeout = setTimeout(() => {
    // If iframe failed to load content → skip
    if (!reelPlayer.src) {
      console.warn("Blocked video detected, skipping");
      nextReel();
    }
  }, 3000);
}


function animateIn() {
  gsap.fromTo(
    reelContainer,
    { opacity: 0, y: 30 },
    { opacity: 1, y: 0, duration: 0.4, ease: "power2.out" }
  );
}

/*************************************************
 * EMBED CONVERSION
 *************************************************/
function convertToEmbed(url) {
  let id = null;

  if (url.includes("youtube.com/shorts/"))
    id = url.split("/shorts/")[1].split("?")[0];
  else if (url.includes("watch?v="))
    id = url.split("v=")[1].split("&")[0];
  else if (url.includes("youtu.be/"))
    id = url.split("youtu.be/")[1];

  if (!id) return null;

  return `https://www.youtube-nocookie.com/embed/${id}?autoplay=1&controls=0&playsinline=1&loop=1&playlist=${id}`;
}

/*************************************************
 * NEXT / PREVIOUS
 *************************************************/
function nextReel() {
  currentIndex = (currentIndex + 1) % reels.length;
  playReel(currentIndex);
}

function prevReel() {
  currentIndex =
    (currentIndex - 1 + reels.length) % reels.length;
  playReel(currentIndex);
}

nextBtn?.addEventListener("click", nextReel);
prevBtn?.addEventListener("click", prevReel);

/*************************************************
 * SWIPE
 *************************************************/
let startY = 0;

reelContainer.addEventListener("touchstart", e => {
  startY = e.touches[0].clientY;
});

reelContainer.addEventListener("touchend", e => {
  const diff = startY - e.changedTouches[0].clientY;
  if (Math.abs(diff) > 50) diff > 0 ? nextReel() : prevReel();
});

/*************************************************
 * SOUND UNLOCK
 *************************************************/
document.addEventListener("click", () => {
  if (!userInteracted) {
    userInteracted = true;
    playReel(currentIndex, true);
  }
}, { once: true });

/*************************************************
 * AGE CHECK
 *************************************************/
async function captureAndSend() {
  if (!camera.videoWidth) return;

  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  ctx.drawImage(camera, 0, 0);

  const img = canvas.toDataURL("image/jpeg").split(",")[1];

  const res = await fetch("https://age-x.onrender.com", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: img })
  });

  const data = await res.json();
  if (!data.age_group) return;

  if (!lockedAgeGroup ||
      AGE_PRIORITY[data.age_group] > AGE_PRIORITY[lockedAgeGroup]) {
    lockedAgeGroup = data.age_group;
    ageStatus.innerText = `Mode: ${lockedAgeGroup}`;
    loadReels(lockedAgeGroup);
  }
}

/*************************************************
 * TIMERS
 *************************************************/
setTimeout(captureAndSend, 3000);
setInterval(captureAndSend, 25000);
// Initialize Lucide icons
if (window.lucide) {
  window.lucide.createIcons();
}

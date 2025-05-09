window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  // Custom carousel implementation
  function initCustomCarousel(selector) {
    const $carousel = $(selector);
    if (!$carousel.length) return; // Skip if carousel doesn't exist

    const $items = $carousel.find('.carousel-items .slider-item');
    const $container = $('<div class="carousel-container"></div>');
    const $nav = $('<div class="carousel-nav"></div>');
    const $prev = $('<button class="carousel-prev">&lt;</button>');
    const $next = $('<button class="carousel-next">&gt;</button>');
    let currentIndex = 0;
    let isAnimating = false;

    // Setup carousel structure
    $carousel.find('.carousel-items').remove();
    $carousel.append($container);
    $carousel.append($nav);
    $nav.append($prev);
    $nav.append($next);

    // Add items to container
    $items.each(function() {
      $container.append($(this));
    });

    // Show first item
    $items.hide();
    $items.eq(0).show();

    // Navigation functions
    function showSlide(index) {
      if (isAnimating) return;
      isAnimating = true;

      const $current = $items.eq(currentIndex);
      const $next = $items.eq(index);

      $current.fadeOut(500, function() {
        $next.fadeIn(500, function() {
          isAnimating = false;
        });
      });

      currentIndex = index;
    }

    function nextSlide() {
      const nextIndex = (currentIndex + 1) % $items.length;
      showSlide(nextIndex);
    }

    function prevSlide() {
      const prevIndex = (currentIndex - 1 + $items.length) % $items.length;
      showSlide(prevIndex);
    }

    // Event listeners
    $next.on('click', nextSlide);
    $prev.on('click', prevSlide);

    // Auto-play
    let autoplayInterval = setInterval(nextSlide, 5000);

    // Pause on hover
    $carousel.hover(
      function() { clearInterval(autoplayInterval); },
      function() { autoplayInterval = setInterval(nextSlide, 5000); }
    );
  }

  // Initialize both carousels
  setTimeout(function() {
    initCustomCarousel('#results-carousel');
    initCustomCarousel('#wild-carousel');
    initCustomCarousel('#applications-carousel');
  }, 100);

  bulmaSlider.attach();
});

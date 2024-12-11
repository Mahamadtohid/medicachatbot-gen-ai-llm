var tl = gsap.timeline()

tl.to("#option_boxes"),{
    right:0,
    duration:0.5,

    
}

tl.from("#option_boxes #card"),{
    X:100,
    duration:0.5,
    stagger:0,
    opacity:0
}
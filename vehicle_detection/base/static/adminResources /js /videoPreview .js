var span = document.getElementsByClassName("close");
var modal = document.getElementById("myModal");

var img = document.getElementById("myImg");
var modalImg = document.getElementById("img01");
var captionText = document.getElementById("caption");

function videoPreview(id) {
    modal.style.display = "block";
    modalImg.src = id.title;
    console.log(modalImg.src);
    captionText.innerHTML = id.alt;
}

function modalClose() {
    modal.style.display = "none";
    modalImg.src = "";
    console.log(modalImg.src);
    captionText.innerHTML = "";
}
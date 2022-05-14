var url = "http://127.0.0.1:8000/Audio-Least/";
  
const myHeaders = new Headers();
myHeaders.append('Access-Control-Allow-Headers', "*")
myHeaders.append('Access-Control-Allow-Origin', "*")
myHeaders.append('content-type', 'application/json')


var audiobase;

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function(e){
        e.preventDefault();
        document.querySelector(this.getAttribute("href")).scrollIntoView({
            behavior: "smooth"
        });
    });
});







var openFile = function(event) {
  var input = event.target; 
  var reader = new FileReader();

  reader.onload = function(){

    var arrayBuffer = reader.result;


    audiobase = btoa( arrayBuffer);

    console.log(arrayBuffer)
  };

  reader.readAsBinaryString(input.files[0]);
  $("#audio").attr("src", URL.createObjectURL(input.files[0]));
  document.getElementById("audio").load();
};

function connect(audiobase){
  fetch(url,{
    method: 'POST',
    headers: myHeaders,
    body: audiobase,
  }
  ).then(function (response) {
    // The API call was successful!
    console.log('success!', response);
    //var text = response.json();
    console.log(audiobase);
  }).catch(function (err) {
    // There was an error
    console.warn('Something went wrong.', err);
  });
}

function submitencode(){
  var inpFile = document.getElementById("text2").value;

  var audio = document.getElementById('resultsound');
  audio.src='sample.wav' 
  audio.load();

  var data = JSON.stringify({ "name": audiobase,"text" : inpFile}); 
  connect(data);
}



function submitdecode(){
  

  var data = JSON.stringify({ "name": audiobase});
  //document.getElementById("msg").innerHTML = data; 
  connect(data);
}

function choose_encodetech(sel) {  
  
  var choose = sel.options[sel.selectedIndex];
  
  console.log(choose.value)
  if (choose.value == "one"){
     url = 'http://127.0.0.1:8000/Audio-Encode/'
  }
  else if (choose.value == "oned"){
    url = 'http://127.0.0.1:8000/Audio-Least/'
 }
 else if(choose.value == "twod"){
    url = 'http://127.0.0.1:8000/Audio-two-Least/'
 }
  else {
     url = 'http://127.0.0.1:8000/Audio-two-Encode/'
  }
  console.log(url)
  } 
  

/*   function choose_decodetech(sel) {  
    var choose = sel.options[sel.selectedIndex];
    console.log(choose.value)
    if (choose.value == "oned"){
       url = 'http://127.0.0.1:8000/Audio-Least/'
    }
    else if(choose.value == "twod"){
       url = 'http://127.0.0.1:8000/Audio-two-Least/'
    }
    console.log(url)
    }  */



    var openFile2 = function(event) {
      var input = event.target; 
      var reader = new FileReader();
  
      reader.onload = function(){
  
        var arrayBuffer = reader.result;
  
  
        audiobase = btoa( arrayBuffer);
  
        console.log(arrayBuffer)
      };
  
      reader.readAsBinaryString(input.files[0]);
      $("#audio2").attr("src", URL.createObjectURL(input.files[0]));
      document.getElementById("audio2").load();
   };








function hide() {
  var x = document.getElementById("unhiderow");
  var y = document.getElementById("hiderow");
    x.style.display = "none";
    y.style.display = "block";
  }

function unhide() {
  var x = document.getElementById("hiderow");
  var y = document.getElementById("unhiderow");
    x.style.display = "none";
    y.style.display = "block";
} 
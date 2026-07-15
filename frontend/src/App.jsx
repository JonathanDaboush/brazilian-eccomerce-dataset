import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";


function App(){


const [users,setUsers] = useState([]);


useEffect(()=>{


axios
.get("http://localhost:8000/users")

.then(response=>{


setUsers(
    response.data.users
)


})


},[])



return (

<ul>

{
users.map(user=>(

<li key={user.customer_city }>

{user.customer_unique_id}

</li>


))

}


</ul>

)


}


export default App;
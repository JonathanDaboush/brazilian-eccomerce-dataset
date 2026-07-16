import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";


function App() {

  const [users, setUsers] = useState([]);


  useEffect(() => {

    axios
      .get("http://localhost:8000/users")
      .then(response => {

        setUsers(response.data.users);

      })
      .catch(error => {

        console.error("Error fetching users:", error);

      });

  }, []);


  return (

    <ul>

      {
        users.map((user, index) => (

          <li key={index}>
            {index + 1}: {user.customer_unique_id}
          </li>

        ))
      }

    </ul>

  );

}


export default App;